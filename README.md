# Financial Health Prediction Challenge

Machine learning competition solution for predicting financial health categories from survey and financial-inclusion data.

## Problem Statement

The goal is to classify individuals or businesses into financial health categories using demographic, behavioral, and financial-access variables. This type of prediction can support financial inclusion research, targeted support programs, and better segmentation for digital finance products.

## Why This Project Matters

Financial inclusion projects often depend on messy real-world survey data with inconsistent category values, missing responses, and imbalanced target classes. This project demonstrates the practical work required before modeling: cleaning, feature engineering, robust encoding, cross-validation, and model ensembling.

## Key Features

- Data cleaning for inconsistent "do not know" and missing-value responses.
- Feature engineering for income, expenses, insurance, banking, and mobile-money behavior.
- Multi-class classification with XGBoost, LightGBM, and CatBoost.
- Stratified cross-validation and macro-F1 evaluation.
- Ensemble weight search for blended predictions.
- Submission generation for competition-style workflows.

## Tech Stack

- Python
- pandas and NumPy
- scikit-learn
- XGBoost
- LightGBM
- CatBoost
- Jupyter Notebook

## Dataset Files

| File | Purpose |
|---|---|
| `Train.csv` | Training data with target labels |
| `Test.csv` | Test data for prediction |
| `SampleSubmission.csv` | Submission format |
| `VariableDefinitions.csv` | Column descriptions |
| `submission.csv` | Generated model submission |

## Project Structure

```text
.
├── solution.ipynb
├── Starter Notebook.ipynb
├── fix_notebook.py
├── test_cell5.py
├── test_cells_2_5.py
├── Train.csv
├── Test.csv
├── SampleSubmission.csv
├── VariableDefinitions.csv
├── submission.csv
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Open `solution.ipynb` and run the notebook from top to bottom. The final cells generate class probabilities, optimize ensemble weights, and write `submission.csv`.

## Evaluation

The notebook uses macro-F1 because the target is multi-class and class balance matters. This metric gives each class meaningful weight instead of rewarding a model that only performs well on the largest class.

## Current Limitations

- Generated training logs are still present in the repository from earlier experiment runs.
- The notebook should be re-executed cleanly before being used as a pinned portfolio project.
- A short results section with final macro-F1, leaderboard score, or validation summary should be added after the final confirmed run.

## Future Improvements

- Move reusable cleaning and feature engineering into `src/`.
- Add automated tests for data cleaning and feature generation.
- Add a notebook execution check in GitHub Actions.
- Add a concise competition write-up with final model comparison.

## Author

Maurice Leonard Okurut
