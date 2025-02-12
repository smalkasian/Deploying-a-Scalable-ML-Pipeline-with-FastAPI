import os
import pandas as pd
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    performance_on_categorical_slice,
    save_model,
    train_model,
)
# load the cencus.csv data
project_path = os.getcwd()  # Use current working directory
data_path = os.path.join(project_path, "data", "census.csv")
data = pd.read_csv(data_path)
print(data_path)

# split the provided data to have a train dataset and a test dataset
train, test = train_test_split(data, test_size=0.2, random_state=42)


# DO NOT MODIFY
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# use the process_data function provided to process the data.
X_train, y_train, encoder, lb = process_data(
    train,
    categorical_features=cat_features,
    label="salary",
    training=True
)

X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)

# use the train_model function to train the model on the training dataset
model = train_model(X_train, y_train)

# save the model and the encoder
model_dir = os.path.join(project_path, "model")
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, "model.pkl")
encoder_path = os.path.join(model_dir, "encoder.pkl")

save_model(model, model_path)
save_model(encoder, encoder_path)

# load the model
model = load_model(
    model_path
) 

# use the inference function to run the model inferences on the test dataset.
preds = inference(model, X_test)

# Calculate and print the metrics
p, r, fb = compute_model_metrics(y_test, preds)
print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}")

# compute the performance on model slices using the performance_on_categorical_slice function
slice_output_path = os.path.join(project_path, "slice_output.txt")
with open(slice_output_path, "w") as f:  # Overwrite file
    for col in cat_features:
        for slice_value in sorted(test[col].unique()):
            count = test[test[col] == slice_value].shape[0]
            p, r, fb = performance_on_categorical_slice(
                test, col, slice_value, cat_features, "salary", encoder, lb, model
            )
            if p is not None:  # Avoid writing if the slice was empty
                print(f"{col}: {slice_value}, Count: {count:,}", file=f)
                print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}", file=f)
                print("-" * 40, file=f)

print(f"Model performance on categorical slices saved to {slice_output_path}")