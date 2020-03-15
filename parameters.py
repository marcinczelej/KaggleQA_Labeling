# We want to predict all needed answers in range 0 - 1 so 30 answers in range 0 -1 
num_labels = 30
epochs = 30
batch_size=4
lr = 2e-5
decay = 0.01
warmup_steps = 200
gradient_accumulate_steps=2
n_splits = 5
save_dir = "./checkpoints"
csv_save_dir = "./dataframes"

train_columns = ['question_title', 'question_body', 'answer']

target_columns = ["question_asker_intent_understanding",
    "question_body_critical",
    "question_conversational",
    "question_expect_short_answer",
    "question_fact_seeking",
    "question_has_commonly_accepted_answer",
    "question_interestingness_others",
    "question_interestingness_self",
    "question_multi_intent",
    "question_not_really_a_question",
    "question_opinion_seeking",
    "question_type_choice",
    "question_type_compare",
    "question_type_consequence",
    "question_type_definition",
    "question_type_entity",
    "question_type_instructions",
    "question_type_procedure",
    "question_type_reason_explanation",
    "question_type_spelling",
    "question_well_written",
    "answer_helpful",
    "answer_level_of_information",
    "answer_plausible",
    "answer_relevance",
    "answer_satisfaction",
    "answer_type_instructions",
    "answer_type_procedure",
    "answer_type_reason_explanation",
    "answer_well_written"]