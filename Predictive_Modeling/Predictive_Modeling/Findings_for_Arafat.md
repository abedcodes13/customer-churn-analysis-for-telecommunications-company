# Predictive modelling findings

Test performance: accuracy: 69.46%; precision: 45.71%; recall: 80.45%; f1: 58.30%. These are test results for a configuration and threshold selected using validation only, on 1349 test records.

The model identified 288 of 358 actual churners, missed 70, and generated 342 false alerts. It correctly classified 649 non-churners. The always-no-churn baseline accuracy is 73.46%; that baseline identifies no churners.

Selected model: weighted_regularised_32_16, hidden layers [32, 16], sigmoid output, threshold 0.45, seed 42, restored epoch 41. Selection used mean validation F1 across three seeds. See validation_threshold_search.csv for all comparisons; do not interpret seed variation as cross-validation.

Largest positive grouped validation AP decreases: Tenure, charges and derived spending, Phone and multiple lines, Senior citizen. These are groups the model relies on, not proof of causal drivers or direction of influence. Correlated engineered features were grouped, so their individual effects cannot be separated here.

Development profiles in development_customer_profiles.csv describe observed churn rates by tenure band and encoded contract group. Read them alongside grouped importance and Abed's independent clustering findings. The reference contract label must be verified against Tyler's encoding before naming it in the report.

Recommendation for team discussion: use the score to prioritise a small retention pilot within a defined contact budget. Review false-positive workload and missed churners before operational use. Compare targeted interventions with a control group; no retention uplift or financial benefit has been demonstrated by this model.

Additional recommendation: consider offers or service reviews only where the observed profiles and business context support them. Feature importance alone does not justify a specific discount or prove why customers left. Gender and age-related inputs should not automatically determine offers; review suitability and performance across relevant groups before deployment.

Limitations: one reduced historical dataset, inherited deduplication without customer IDs, a single fixed train/validation/test split, possible validation selection optimism, correlated features, and uncalibrated sigmoid scores. No temporal/external validation or causal evaluation. Bootstrap intervals quantify test-sample uncertainty for this fixed model only.

Proposed improvements: confirm the duplicate policy and reference categories with Tyler; collect richer and time-stamped features; evaluate on future data; assess calibration and business costs; monitor drift and relevant subgroup performance. These are future actions, not completed checks.

Handoff status: review release. Arafat can use these findings and figures to draft the predictive-modelling sections. The team still needs to review wording, integrate segmentation results, complete the final report and record the 10–15 minute group demonstration. PM submits the team package.