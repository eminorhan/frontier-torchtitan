from datasets import load_dataset

ds = load_dataset("hails/mmlu_no_train", "all", trust_remote_code=True)

# ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", trust_remote_code=True)
# ds = load_dataset("allenai/ai2_arc", "ARC-Easy", trust_remote_code=True)

# ds = load_dataset("openai/gsm8k", "main", trust_remote_code=True)
# ds = load_dataset("openai/gsm8k", "socratic", trust_remote_code=True)

# bbh_subsets = ['boolean_expressions', 'causal_judgement', 'date_understanding', 'disambiguation_qa', 'dyck_languages', 'formal_fallacies', 'geometric_shapes', 'hyperbaton', 'logical_deduction_five_objects', 'logical_deduction_seven_objects', 'logical_deduction_three_objects', 'movie_recommendation', 'multistep_arithmetic_two', 'navigate', 'object_counting', 'penguins_in_a_table', 'reasoning_about_colored_objects', 'ruin_names', 'salient_translation_error_detection', 'snarks', 'sports_understanding', 'temporal_sequences', 'tracking_shuffled_objects_five_objects', 'tracking_shuffled_objects_seven_objects', 'tracking_shuffled_objects_three_objects', 'web_of_lies', 'word_sorting']
# for bbh_subset in bbh_subsets:
#     ds = load_dataset("lukaemon/bbh", bbh_subset, trust_remote_code=True)

# winogrande_subsets = ['winogrande_xs', 'winogrande_s', 'winogrande_m', 'winogrande_l', 'winogrande_xl', 'winogrande_debiased']
# for winogrande_subset in winogrande_subsets:
#     ds = load_dataset("allenai/winogrande", winogrande_subset, trust_remote_code=True)

# ds = load_dataset("Rowan/hellaswag", trust_remote_code=True)