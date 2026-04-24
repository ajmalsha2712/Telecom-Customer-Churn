<<<<<<< HEAD
# Telecom Churn Prediction (pandas + Streamlit)

This folder contains:

- `train.py`: trains a full scikit-learn pipeline (preprocessing + model) using `Telco-Customer-Churn.csv`
- `app.py`: Streamlit UI for single + batch churn prediction, plus evaluation/retraining

## Setup

```bash
pip install -r requirements.txt
```

## Train the model

From this folder:

```bash
python train.py --data "Telco-Customer-Churn.csv" --out "customer_churn_model.pkl"
```

## Run the Streamlit app

```bash
streamlit run app.py
```

## Batch prediction format

Upload a CSV that has the same feature columns as the dataset. It can include `customerID` and/or `Churn` (the app will ignore them for prediction).

=======
# Telecom Churn Prediction (pandas + Streamlit)

This folder contains:

- `train.py`: trains a full scikit-learn pipeline (preprocessing + model) using `Telco-Customer-Churn.csv`
- `app.py`: Streamlit UI for single + batch churn prediction, plus evaluation/retraining

## Setup

```bash
pip install -r requirements.txt
```

## Train the model

From this folder:

```bash
python train.py --data "Telco-Customer-Churn.csv" --out "customer_churn_model.pkl"
```

## Run the Streamlit app

```bash
streamlit run app.py
```

## Batch prediction format

Upload a CSV that has the same feature columns as the dataset. It can include `customerID` and/or `Churn` (the app will ignore them for prediction).

>>>>>>> 60b8d1da7275ecfe548860d2249a3e0e853f15a1
