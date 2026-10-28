# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This project uses a Random Forest Classifier to predict whether an individual's annual income is greater than $50,000 based on demographic and employment-related information from the Census Income dataset. The model is implemented using Python and scikit-learn. It uses 100 decision trees and a random state of 42 to support reproducible training. The data processing pipeline uses one-hot encoding for categorical features and label binarization for the income target. The model is part of an educational machine learning project that includes a FastAPI application for inference.

## Intended Use

The model is intended for educational and demonstration purposes. It demonstrates how to process tabular data, train a binary classification model, evaluate predictions, and serve predictions through an API. Its intended users are students, developers, and researchers learning about machine learning pipelines.

The model should not be used to make employment, lending, insurance, housing, or other consequential decisions about individuals. It has not been validated for real-world decision-making.

## Training Data

The model was trained using the `data/census.csv` dataset included in this project. The dataset contains demographic, education, employment, and related information, with the `salary` column serving as the prediction target. The target distinguishes individuals earning more than $50,000 annually from those earning $50,000 or less.

The data was split into training and testing sets using an 80/20 split, a random state of 42, and stratification by the `salary` target. The training data was processed using one-hot encoding for categorical features and label binarization for the target. The categorical features include workclass, education, marital status, occupation, relationship, race, sex, and native country.

## Evaluation Data

The model was evaluated on the 20% test partition held out from `data/census.csv`. The split used a random state of 42 and stratification by the `salary` column to preserve the target-class proportions.

The test data was transformed using the encoder and label binarizer fitted on the training data. The model generated predictions for the test set, and performance was measured using precision, recall, and F1 score. Additional evaluation was performed across individual values of each categorical feature, with the slice-level results written to `slice_output.txt`.

## Metrics

The model's test-set performance was:

- **Precision: 0.7353 (73.53%).** Of the individuals predicted to earn more than $50,000, approximately 73.53% belonged to the positive class.
- **Recall: 0.6378 (63.78%).** The model correctly identified approximately 63.78% of individuals in the positive class.
- **F1 score: 0.6831 (68.31%).** This is the harmonic mean of precision and recall and summarizes the balance between the two metrics.

These results indicate that the model identifies a majority of the positive-class examples, but it also misses some individuals who belong to that class. The metrics describe performance on this particular held-out test set and do not establish performance on other populations or in real-world settings.

## Ethical Considerations

The dataset contains sensitive or potentially sensitive attributes, including age, sex, race, and native country. Predictions about income can reflect historical inequalities and patterns present in the underlying data. A model trained on these data may reproduce or amplify existing biases, and its performance may differ across demographic groups.

The model should not be used to determine an individual's worth, ability, eligibility, or access to opportunities. Before any higher-impact use could be considered, additional fairness evaluations, privacy reviews, domain expertise, and appropriate legal and ethical assessments would be necessary.

## Caveats and Recommendations

The model is an educational prototype and has not been validated for production use. Its evaluation metrics are based on one random train/test split and may not represent performance on new or changing populations. The reported precision, recall, and F1 score do not by themselves measure fairness, calibration, or the consequences of incorrect predictions.

Users should review the results in `slice_output.txt` to understand how performance varies across categorical groups. Further work should include additional validation, confusion-matrix analysis, subgroup fairness assessments, and testing on appropriately sourced and more recent data. Predictions should not be treated as definitive statements about an individual's income.
