import pickle
import logging

import pandas
import numpy

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier


logger = logging.getLogger(__name__)
logging.basicConfig(level="INFO")


def begin() -> None:
    """
    A function to load the trained model artifact (.pkl) as a global variable.
    The model will be used by other functions to produce predictions.
    The function also sets global variables for feature lists.
    """

    global model
    model = pickle.load(open("./binaries/RFC_model.pkl", "rb"))
    logger.info("'RFC_model.pkl' file loaded to global variable 'model'")

    global numeric_predictors, categorical_predictors, target_variable
    numeric_predictors = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
    categorical_predictors = ["Sex", "Cabin", "Embarked"]
    target_variable = "Survived"
    logger.info("Variable roles assigned")


def predict(scoring_data: dict) -> dict:
    """
    A function to predict Survival on Titanic, given passanger info.
    Args:
        data (dict): input dictionary to be scored, containing predictive features.
    Returns:
        (dict): Scored (predicted) input data.
    """

    scoring_data = pandas.DataFrame([scoring_data])

    scoring_data["Prediction"] = model.predict(
        scoring_data[numeric_predictors + categorical_predictors]
    )
    return scoring_data.to_dict(orient="records")[0]


def metrics(metrics_df: pandas.DataFrame) -> dict:
    """
    A function to compute Accuracy scored and labeled data.
    Args:
        data (pandas.DataFrame): Dataframe of passenger info, including ground truths, predictions.
    Returns:
        (dict): Model accuracy.
    """

    logger.info("metrics_df is of shape: %s", metrics_df.shape)

    X_test = metrics_df.drop("Survived", axis=1)
    y_true = metrics_df["Survived"]
    return {
        "ACCURACY": model.score(
            X_test[numeric_predictors + categorical_predictors], y_true
        )
    }


def train(training_df: pandas.DataFrame) -> None:
    """
    A function to train a random forest classifier on labeled titanic data. Function
    does not return an output, but rather writes trained model to /binaries/.
    Args:
        data (pandas.DataFrame): Dataframe of passenger info, including ground truths.
    """

    logger.info("train_df is of shape: %s", training_df.shape)

    numeric_predictors = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
    categorical_predictors = ["Sex", "Cabin", "Embarked"]
    target_variable = "Survived"

    training_df = training_df.loc[
        :, numeric_predictors + categorical_predictors + [target_variable]
    ]

    logger.info("Replacing Nulls")
    training_df.replace(to_replace=[None], value=numpy.nan, inplace=True)
    training_df[numeric_predictors] = training_df.loc[:, numeric_predictors].apply(
        pandas.to_numeric, errors="coerce"
    )

    logger.info("Setting 'y_train' and 'X_train'")
    X_train = training_df.drop("Survived", axis=1)
    y_train = training_df["Survived"]

    logger.info("Setting up numeric transformer Pipeline")
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ]
    )

    logger.info("Setting up categorical transformer Pipeline")
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    logger.info("Initializing preprocessor")
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_predictors),
            ("cat", categorical_transformer, categorical_predictors),
        ]
    )

    logger.info("Initializing model pipeline")
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(random_state=42)),
        ]
    )

    logger.info("Fitting model")
    model.fit(X_train, y_train)

    # pickle file should be written to /binaries/
    logger.info("Model fitting complete. Writing 'RFC_model.pkl' to ./binaries/")
    with open("./binaries/RFC_model.pkl", "wb") as f:
        pickle.dump(model, f)

    logger.info("Training Job Complete!")


# Test Script
if __name__ == "__main__":
    # Load model
    begin()

    # Loading datasets to test functions
    predict_data = pandas.read_csv("./data/predict.csv")
    test_data = pandas.read_csv("./data/test.csv")
    training_data = pandas.read_csv("./data/train.csv")

    # Function calls
    print(predict(predict_data.iloc[0]))
    print(metrics(test_data))
    train(training_data)
