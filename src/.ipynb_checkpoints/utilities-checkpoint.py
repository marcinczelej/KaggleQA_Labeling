import tensorflow as tf
import numpy as np

"""
    Gradient accumulation implementation
"""

def accumulated_gradients(gradients,
                          step_gradients,
                          num_grad_accumulates) -> tf.Tensor:
    if gradients is None:
        gradients = [flat_gradients(g) / num_grad_accumulates for g in step_gradients]
    else:
        for i, g in enumerate(step_gradients):
            gradients[i] += flat_gradients(g) / num_grad_accumulates
    
    return gradients

# This is needed for tf.gather like operations.
def flat_gradients(grads_or_idx_slices: tf.Tensor) -> tf.Tensor:
    '''Convert gradients if it's tf.IndexedSlices.
    When computing gradients for operation concerning `tf.gather`, the type of gradients 
    '''
    if type(grads_or_idx_slices) == tf.IndexedSlices:
        return tf.scatter_nd(
            tf.expand_dims(grads_or_idx_slices.indices, 1),
            grads_or_idx_slices.values,
            grads_or_idx_slices.dense_shape
        )
    return grads_or_idx_slices


"""
    Custom scheduler with learning rate rising for warmup period and going down later on
"""

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, warmup_steps, num_steps, base_lr):
    super(CustomSchedule, self).__init__()

    self.warmup_steps = tf.cast(warmup_steps, tf.float32)
    self.num_steps = tf.cast(num_steps, tf.float32)
    self.lr = tf.cast(base_lr, tf.float32)

  def __call__(self, step):
    def warmupPhase() : return step/tf.math.maximum(1.0, self.warmup_steps)
    def decayPhase() : return tf.math.maximum(0.0, (self.num_steps - step))/tf.math.maximum(1.0, self.num_steps - self.warmup_steps)

    multiplier = tf.cond(tf.math.less(step, self.warmup_steps), warmupPhase, decayPhase)
    
    return self.lr * multiplier

class CosineDecayWithWarmup(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, warmup_steps, total_steps, base_lr, hold_base_rate_steps=0):
        super(CosineDecayWithWarmup, self).__init__()

        self.warmup_steps = tf.cast(warmup_steps, tf.float32)
        self.total_steps = tf.cast(total_steps, tf.float32)
        self.base_lr = tf.cast(base_lr, tf.float32)
        self.hold_base_rate_steps = tf.cast(hold_base_rate_steps, tf.float32)
        
        if total_steps < warmup_steps:
            raise ValueError('total_steps must be larger or equal to '
                     'warmup_steps.')
    
    def eager_decay_rate(self, global_step):
        """Callable to compute the learning rate."""
        warmup_learning_rate = float(global_step) / float(max(1, self.warmup_steps))
        learning_rate = 0.5 * self.base_lr * (1 + tf.cos(np.pi * (tf.cast(global_step, tf.float32) - self.warmup_steps - self.hold_base_rate_steps) / float(self.total_steps - self.warmup_steps - self.hold_base_rate_steps)))
        if self.hold_base_rate_steps > 0:
          learning_rate = tf.where(
              global_step > self.warmup_steps + self.hold_base_rate_steps,
              learning_rate, self.base_lr)
        if self.warmup_steps > 0:
          if self.base_lr < warmup_learning_rate:
            warmup_learning_rate = self.base_lr
          slope = (self.base_lr - warmup_learning_rate) / self.warmup_steps
          warmup_rate = slope * tf.cast(global_step,
                                        tf.float32) + warmup_learning_rate
          learning_rate = tf.where(global_step < self.warmup_steps, warmup_rate,
                                   learning_rate)
        return tf.where(global_step > self.total_steps, 0.0, learning_rate,
                        name='learning_rate')
    
    def __call__(self, step):
        return self.eager_decay_rate(step)
        

def cosine_decay_with_warmup(global_step,
                             learning_rate_base,
                             total_steps,
                             warmup_learning_rate=0.0,
                             warmup_steps=0,
                             hold_base_rate_steps=0):
  """Cosine decay schedule with warm up period.
  Cosine annealing learning rate as described in:
    Loshchilov and Hutter, SGDR: Stochastic Gradient Descent with Warm Restarts.
    ICLR 2017. https://arxiv.org/abs/1608.03983
  In this schedule, the learning rate grows linearly from warmup_learning_rate
  to learning_rate_base for warmup_steps, then transitions to a cosine decay
  schedule.
  Args:
    global_step: int64 (scalar) tensor representing global step.
    learning_rate_base: base learning rate.
    total_steps: total number of training steps.
    warmup_learning_rate: initial learning rate for warm up.
    warmup_steps: number of warmup steps.
    hold_base_rate_steps: Optional number of steps to hold base learning rate
      before decaying.
  Returns:
    If executing eagerly:
      returns a no-arg callable that outputs the (scalar)
      float tensor learning rate given the current value of global_step.
    If in a graph:
      immediately returns a (scalar) float tensor representing learning rate.
  Raises:
    ValueError: if warmup_learning_rate is larger than learning_rate_base,
      or if warmup_steps is larger than total_steps.
  """
  if total_steps < warmup_steps:
    raise ValueError('total_steps must be larger or equal to '
                     'warmup_steps.')
  def eager_decay_rate():
    """Callable to compute the learning rate."""
    learning_rate = 0.5 * learning_rate_base * (1 + tf.cos(
        np.pi *
        (tf.cast(global_step, tf.float32) - warmup_steps - hold_base_rate_steps
        ) / float(total_steps - warmup_steps - hold_base_rate_steps)))
    if hold_base_rate_steps > 0:
      learning_rate = tf.where(
          global_step > warmup_steps + hold_base_rate_steps,
          learning_rate, learning_rate_base)
    if warmup_steps > 0:
      if learning_rate_base < warmup_learning_rate:
        raise ValueError('learning_rate_base must be larger or equal to '
                         'warmup_learning_rate.')
      slope = (learning_rate_base - warmup_learning_rate) / warmup_steps
      warmup_rate = slope * tf.cast(global_step,
                                    tf.float32) + warmup_learning_rate
      learning_rate = tf.where(global_step < warmup_steps, warmup_rate,
                               learning_rate)
    return tf.where(global_step > total_steps, 0.0, learning_rate,
                    name='learning_rate')

  if tf.executing_eagerly():
    return eager_decay_rate
  else:
    return eager_decay_rate()