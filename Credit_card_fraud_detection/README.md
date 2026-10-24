# Credit Card Fraud Detection

## Project Overview
Credit Card Fraud Detection is a Machine Learning project developed as part of my Machine Learning Internship at SAM AI Technologies. The objective is to identify potentially fraudulent credit card transactions using classification algorithms and evaluate their performance.

## Objectives
- Analyze credit card transaction data.
- Identify fraudulent and legitimate transactions.
- Train and compare Logistic Regression and Random Forest models.
- Handle imbalanced transaction data using class weights.
- Evaluate models using Precision, Recall, F1-score, ROC-AUC, and a confusion matrix.
- Build an interactive dashboard using Streamlit.

## Technologies Used
- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Streamlit
- Joblib

## Dataset
The dataset contains transaction features including:
- `Time`
- `V1` to `V28`
- `Amount`
- `Class` — target variable

Target values:
- `0` — Legitimate transaction
- `1` — Potentially fraudulent transaction

Dataset source: [Credit Card Fraud Detection — Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)


## Installation

1. Clone or download this repository.
2. Open the project folder in VS Code.
3. Create and activate a virtual environment.

```bash
python -m venv venv
venv\Scripts\activate
```

4. Install the required libraries.

```bash
pip install -r requirements.txt
```

## Run the Application

Start the Streamlit dashboard:

```bash
python -m streamlit run app.py
```

Open the local URL shown in the terminal, usually `http://localhost:8501`.

The dashboard provides dataset overview, exploratory analysis, model comparison, and CSV-based transaction predictions.

## Model Evaluation
The project evaluates the models using:
- Precision
- Recall
- F1-score
- ROC-AUC
- Average Precision
- Confusion Matrix

Actual evaluation scores are generated when the models are trained. Results should be interpreted with care because fraudulent transactions represent a small proportion of the dataset.

## Important Notes
- The saved `.pkl` files contain trained model objects and are not intended to be opened as ordinary text files.
- Model artifacts must be generated before using an application that expects them.
- Predictions are estimates and do not prove that a transaction is fraudulent.
- This is an educational project, not a production banking security system.

