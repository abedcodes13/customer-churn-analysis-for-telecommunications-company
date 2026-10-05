# Predictive modelling review release

Run ID (UTC): 20261005T040331Z
Data commit: 6d1cb202e66568114f8d9a34a4cdeeb3dffabed3

## Main files
- ANN_Architecture.pdf: required architecture document.
- ann_final.keras: trained model evaluated on test.
- scaler_parameters.npz and metadata.json: matching preprocessing, feature order and threshold.
- predict_churn.py: inference on encoded unscaled features.
- results/final_test_metrics.csv and results/test_metric_intervals.csv: evaluation and uncertainty.
- results/test_predictions.csv: probabilities and labels; source_row_index is not a customer ID.
- results/validation_runs_at_050.csv and results/validation_threshold_search.csv: model selection evidence.
- results/grouped_feature_importance.csv and results/development_customer_profiles.csv: interpretation evidence.
- figures/: training, threshold, test and interpretation graphics.
- Findings_for_Arafat.md and .pdf: report input for team review.
- requirements.txt: versions used by this run.

## Reproduce and submit
Add the executed ANN_Churn_Prediction_Stage3.ipynb notebook to this folder after downloading it from Colab. Run it top to bottom with the pinned source data. Outputs can vary across platforms and library versions.

The deployed input contract here means Tyler-style encoded unscaled features, not raw categorical customer records and not already-scaled features. No web application or production deployment is supplied.

The selected model was trained on 4,313 rows, with 1,079 validation rows. The 1,349 test rows were used after selection. Do not tune using test outcomes. Candidate model files are retained as audit evidence; ann_final.keras is the selected model.

Before submission: inspect the PDFs and figures, check agreement across report and results, integrate Abed's clustering, have Arafat review findings, and prepare the group video. Put these deliverables in Predictive_Modeling in the existing team repository. The PM submits the group report, repository link and video.
