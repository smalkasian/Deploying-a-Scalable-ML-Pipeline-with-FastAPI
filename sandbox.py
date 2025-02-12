#------------------------------------------------------------------------------------
# Developed by CSamuel Malkasian
#------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized. Please Fork Instead.
#------------------------------------------------------------------------------------

CURRENT_VERSION = "1.0"
CHANGES = """

"""

#------------------------------------NOTES-------------------------------------------
# Used to test code and run misc functions.


#------------------------------------IMPORTS-----------------------------------------
import pandas as pd
import requests
import numpy as np
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import train_model, inference
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

#-------------------------------------MAIN-------------------------------------------

#Used to print metri scores for the model_card.md
def get_metric_scores():
    data_path = "data/census.csv"
    df = pd.read_csv(data_path)

    train, test = train_test_split(df, test_size=0.2, random_state=42)

    # Categorical features
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

    # Process test data
    X_test, y_test, encoder, lb = process_data(
        test, categorical_features=cat_features, label="salary", training=True
    )

    # Load trained model
    model_path = "model/model.pkl"
    model = train_model(X_test, y_test)
    preds = inference(model, X_test)  # Run inference
    y_test = np.array(y_test)

    # Print to verify
    print("y_test shape:", y_test.shape)
    print("preds shape:", preds.shape)
    print("First 5 y_test:", y_test[:5])
    print("First 5 preds:", preds[:5])

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    print(model.get_params())  # Output all hyperparameters
    cm = confusion_matrix(y_test, preds)
    print(cm)
    roc_auc = roc_auc_score(y_test, preds)
    print("ROC-AUC Score:", roc_auc)
