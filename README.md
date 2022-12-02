# titanic_survival_RFC

A Model (Random Forest Classifier) to predict likelihood of Survival on board the ill-fated Titanic.

Model code contains scoring (prediction), metrics (accuracy), and training functions.

## Running Locally

To run this model locally, create a new Python 3.9.8 virtual environment
(such as with `pyenv`). Then, use the following command to update `pip`
and `setuptools`:

```
python3 -m pip install --upgrade setuptools
python3 -m pip install --upgrade pip
```

And install the required libraries:

```
python3 -m pip install -r requirements.txt
```

The main source code is contained in `titanic.py`. To test all code at-once, run

```
python3 titanic.py
```

## Scoring Jobs

### Sample Inputs

Choose the following file for a sample scoring job:
 - `predict.csv`

### Sample Output

The output of the scoring job when the input data is `predict.csv` is a JSONS file (one-line JSON records). Here are the first two output records:
```json
{"Ticket": 349248, "SibSp": 0, "Sex": "male", "Pclass": 3, "PassengerId": 871, "Parch": 0, "Name": "Balkic, Mr. Cerin", "Fare": 7.8958, "Embarked": "S", "Cabin": null, "Age": 26.0, "Prediction": 0}
{"Ticket": 113781, "SibSp": 1, "Sex": "female", "Pclass": 1, "PassengerId": 499, "Parch": 2, "Name": "Allison, Mrs. Hudson J C (Bessie Waldo Daniels)", "Fare": 151.55, "Embarked": "S", "Cabin": "C22 C26", "Age": 25.0, "Prediction": 1}
```

## Metrics Jobs

Model code includes a metrics function used to compute accuracy.

### Sample Inputs

Choose the following file for a sample metrics job:
 - `test.csv`

### Sample Oututs
The output of the metrics job when the input data is `test.csv` is the following JSON record:
```json
{"ACCURACY": 0.8111888111888111}
```

## Training Jobs

Model Code includes a training function used to train a model binary.

### Sample Inputs

Choose **one** of:
 - `train.csv`
 - `train.json`

### Output Files

 - `RFC_model.pkl`
