# Model Card

For additional information see the Model Card paper: [Model Card Paper](https://arxiv.org/pdf/1810.03993.pdf)

## Model Details

- **Algorithm**: `RandomForestClassifier` (100 trees, `random_state=42`)
- **Framework**: `scikit-learn`
- **Preprocessing**: One-hot encoding for categorical variables, label binarization for target
- **Hyperparameters**:
  - `n_estimators`: 100
  - `max_depth`: None (default)
  - `min_samples_split`: 2
  - `min_samples_leaf`: 1
  - `bootstrap`: True

## Intended Use

- Predicting income classification (`<=50K` or `>50K`) based on Census Bureau data.
- Designed for educational purposes in machine learning pipelines and model deployment.
- Not intended for real-world financial or hiring decisions without fairness audits.

## Training Data

- **Dataset**: Census Bureau data (`census.csv`)
- **Size**: `32,561` rows, `15` columns
- **Features**: `9` categorical, `6` numerical
- **Preprocessing**:
  - Categorical features one-hot encoded
  - Continuous features left unscaled (could be improved with normalization)
  - Target (`salary`) binarized (`<=50K` → `0`, `>50K` → `1`)

## Evaluation Data

- **Split**: `80% train`, `20% test`
- **Test Set Size**: `~6,512 samples`
- **Processed identically to training data**
- **Performance Measured on Full Dataset and Categorical Slices**

## Metrics

- **Precision**: `0.7391`
- **Recall**: `0.6384`
- **F1-score**: `0.6851`
- **Categorical Slice Performance**: See `slice_output.txt`
- **Confusion Matrix**: Captures false positives and false negatives (not included in this summary but recommended for review)
- **ROC-AUC Score**: Recommended for evaluation in future iterations

## Ethical Considerations

- **Bias in Training Data**: The dataset may contain biases based on historical income disparities.
- **Fairness**: Consider evaluating fairness metrics across demographic groups.
- **Explainability**: Feature importance scores can help interpret decisions.
- **Privacy**: No personally identifiable information (PII) is included.
- **Mitigation Strategies**: Bias reduction techniques such as re-weighting or adversarial debiasing could be applied.

## Caveats and Recommendations

- **Feature Engineering**: Feature selection and scaling may improve performance.
- **Model Limitations**: The model is a simple classifier; deep learning approaches could yield better results.
- **Deployment**:
  - Continuous monitoring is recommended to ensure real-world performance.
  - Consider logging model predictions for periodic audits.
- **Future Improvements**:
  - Experiment with additional algorithms (e.g., Gradient Boosting, XGBoost)
  - Introduce hyperparameter tuning for optimal performance
  - Evaluate fairness metrics such as demographic parity or equalized odds
