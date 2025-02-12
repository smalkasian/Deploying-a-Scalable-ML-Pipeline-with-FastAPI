# Model Card

For additional information see the Model Card paper: [Model Card Paper](https://arxiv.org/pdf/1810.03993.pdf)

## Model Details

- **Algorithm**: `RandomForestClassifier` (100 trees, `random_state=42`)
- **Framework**: `scikit-learn`
- **Preprocessing**: One-hot encoding for categorical variables, label binarization for target

## Intended Use

- Predicting income classification (`<=50K` or `>50K`) based on Census Bureau data.
- Designed for educational purposes in machine learning pipelines and model deployment.

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

## Metrics

_Please include the metrics used and your model's performance on those metrics._

- **Precision**: `0.7391`
- **Recall**: `0.6384`
- **F1-score**: `0.6851`
- **Categorical Slice Performance**: See `slice_output.txt`

## Ethical Considerations

- **Bias in Training Data**: The dataset may contain biases based on historical income disparities.
- **Fairness**: Consider evaluating fairness metrics across demographic groups.
- **Privacy**: No personally identifiable information (PII) is included.

## Caveats and Recommendations

- **Feature Engineering**: Feature selection and scaling may improve performance.
- **Model Limitations**: The model is a simple classifier; deep learning approaches could yield better results.
- **Deployment**: Continuous monitoring is recommended to ensure real-world performance.
