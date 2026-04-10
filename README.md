# Student Performance Prediction — End-to-End Machine Learning Project

A complete, production-style machine learning project that predicts a student's **math score** based on demographic and academic features. Covers every stage of an ML workflow: data ingestion, preprocessing, model training with hyperparameter tuning, and a **Flask web application** for real-time predictions.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [High-Level Architecture](#high-level-architecture)
- [Dataset](#dataset)
- [Tech Stack & Packages](#tech-stack--packages)
- [Project Structure](#project-structure)
- [Component Breakdown](#component-breakdown)
- [Models Trained](#models-trained)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Web Application](#web-application)
- [How to Clone & Run](#how-to-clone--run)
- [Logging & Exception Handling](#logging--exception-handling)

---

## Problem Statement

Given student attributes — gender, race/ethnicity, parental level of education, lunch type, test preparation course — along with **reading** and **writing** scores, predict the student's **math score**.

- **Type:** Supervised Regression  
- **Metric:** R² (Coefficient of Determination)  
- **Acceptance Threshold:** R² ≥ 0.6  

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Training Pipeline                        │
│                                                                 │
│  ┌──────────────────┐    ┌────────────────────┐    ┌─────────┐ │
│  │  Data Ingestion  │───▶│ Data Transformation│───▶│  Model  │ │
│  │                  │    │                    │    │ Trainer │ │
│  │ - Read stud.csv  │    │ - Impute missing   │    │         │ │
│  │ - 80/20 split    │    │ - Scale numericals │    │ - Train │ │
│  │ - Save to        │    │ - Encode categorics│    │   9     │ │
│  │   artifacts/     │    │ - Save             │    │   models│ │
│  │   train.csv      │    │   preprocessor.pkl │    │ - Grid  │ │
│  │   test.csv       │    │                    │    │   Search│ │
│  │   data.csv       │    │                    │    │ - Save  │ │
│  └──────────────────┘    └────────────────────┘    │  model  │ │
│                                                    │  .pkl   │ │
│                                                    └─────────┘ │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      Prediction Pipeline                        │
│                                                                 │
│   Browser / User                                                │
│       │                                                         │
│       ▼                                                         │
│  ┌──────────┐    ┌──────────────┐    ┌─────────────────────┐   │
│  │  Flask   │───▶│  CustomData  │───▶│   PredictPipeline   │   │
│  │  app.py  │    │              │    │                     │   │
│  │          │    │ Structures   │    │ - load model.pkl    │   │
│  │ POST     │    │ form input   │    │ - load              │   │
│  │ /predict │    │ as DataFrame │    │   preprocessor.pkl  │   │
│  └──────────┘    └──────────────┘    │ - transform input   │   │
│       │                              │ - predict & return  │   │
│       ▼                              └─────────────────────┘   │
│  Render result on home.html  ◀────────────────────────────────  │
└─────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────┐
│              Shared Utilities (src/)          │
│                                              │
│  logger.py    → logs to file + terminal      │
│  exception.py → CustomException with         │
│                 file name + line number       │
│  utils.py     → save_object, load_object,    │
│                 evaluate_models (GridSearchCV)│
└──────────────────────────────────────────────┘
```

---

## Dataset

- **Source:** [Students Performance in Exams — Kaggle](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)
- **File:** `notebook/data/stud.csv`
- **Size:** 1000 rows × 8 columns

### Features

| Feature | Type | Example Values |
|---|---|---|
| gender | Categorical | male, female |
| race_ethnicity | Categorical | group A – group E |
| parental_level_of_education | Categorical | bachelor's degree, master's degree, high school, etc. |
| lunch | Categorical | standard, free/reduced |
| test_preparation_course | Categorical | none, completed |
| reading_score | Numerical (0–100) | 72, 90, 55 |
| writing_score | Numerical (0–100) | 74, 88, 60 |

### Target

| Column | Type | Description |
|---|---|---|
| math_score | Numerical (0–100) | Score to predict |

---

## Tech Stack & Packages

### Core
| Package | Purpose |
|---|---|
| `flask` | Web framework for the prediction UI |
| `pandas` | Data loading and DataFrame operations |
| `numpy` | Numerical computations |

### Machine Learning
| Package | Purpose |
|---|---|
| `scikit-learn` | Preprocessing pipelines, models, GridSearchCV, metrics |
| `xgboost` | XGBoost gradient boosting regressor |
| `catboost` | CatBoost gradient boosting regressor |

### Visualization (Notebooks only)
| Package | Purpose |
|---|---|
| `matplotlib` | Plotting graphs |
| `seaborn` | Statistical data visualization |

### Utilities
| Package | Purpose |
|---|---|
| `pickle` | Serializing/deserializing model and preprocessor objects |
| `logging` | Centralized application logging |
| `sys` | Exception traceback details |
| `os` | File and directory path management |
| `setuptools` | Package discovery via `setup.py` |

> Full list in `requirements.txt`

---

## Project Structure

```
ML Project/
│
├── app.py                             # Flask app — routes & prediction endpoint
│
├── templates/                         # Jinja2 HTML templates
│   ├── index.html                     # Landing page
│   └── home.html                      # Prediction form & result display
│
├── artifacts/                         # Auto-generated by the training pipeline
│   ├── data.csv                       # Full raw dataset copy
│   ├── train.csv                      # 80% training split
│   ├── test.csv                       # 20% testing split
│   ├── preprocessor.pkl               # Fitted ColumnTransformer pipeline
│   └── model.pkl                      # Best trained & tuned model
│
├── logs/                              # Auto-generated timestamped log files
│   └── log_MM-DD-YYYY_HH-MM-SS.log
│
├── catboost_info/                     # Auto-generated by CatBoost (gitignored)
│
├── notebook/                          # EDA & experimental notebooks
│   ├── 1.EDA sudent performance.ipynb
│   ├── 2.model_training.ipynb
│   └── data/
│       └── stud.csv                   # Raw source dataset
│
├── src/                               # Main Python package
│   ├── __init__.py
│   ├── exception.py                   # CustomException: script name + line number
│   ├── logger.py                      # Dual logging: file + terminal
│   ├── utils.py                       # save_object, load_object, evaluate_models
│   │
│   ├── components/                    # Core ML pipeline stages
│   │   ├── __init__.py
│   │   ├── data_ingestion.py          # Reads CSV, produces train/test splits
│   │   ├── data_transformation.py     # Builds & fits preprocessing pipeline
│   │   └── model_trainer.py           # Trains 9 models, tunes via GridSearchCV
│   │
│   └── pipeline/
│       ├── __init__.py
│       ├── predict_pipeline.py        # Loads artifacts, runs inference
│       └── train_pipeline.py          # (reserved for orchestrated training)
│
├── mlproject.egg-info/                # Auto-generated by setup.py
├── requirements.txt                   # Project dependencies
├── setup.py                           # Registers src/ as the mlproject package
└── README.md
```

---

## Component Breakdown

### `data_ingestion.py`
- Resolves `stud.csv` path dynamically using `os.path.abspath` relative to project root
- Saves a full raw copy to `artifacts/data.csv`
- Performs an 80/20 train-test split (random state = 42)
- Saves `artifacts/train.csv` and `artifacts/test.csv`
- When run as `__main__`, automatically triggers transformation and model training

### `data_transformation.py`
Constructs a `ColumnTransformer` with two sub-pipelines:

| Pipeline | Columns | Steps |
|---|---|---|
| Numerical | `reading_score`, `writing_score` | `SimpleImputer(median)` → `StandardScaler` |
| Categorical | gender, race, education, lunch, test_prep | `SimpleImputer(most_frequent)` → `OneHotEncoder` → `StandardScaler(with_mean=False)` |

- Saves fitted preprocessor to `artifacts/preprocessor.pkl`
- Returns transformed NumPy arrays for train and test sets

### `model_trainer.py`
- Trains 9 regression models, each tuned with `GridSearchCV` (3-fold CV, `n_jobs=-1`)
- Selects the model with the highest R² on the test set
- Raises `CustomException` if no model achieves R² ≥ 0.6
- Saves the best model to `artifacts/model.pkl`

### `predict_pipeline.py`
- `PredictPipeline.predict(features)` — loads `model.pkl` and `preprocessor.pkl`, transforms the input DataFrame, returns the predicted score
- `CustomData` — accepts 7 user inputs and converts them to a single-row Pandas DataFrame via `get_data_as_data_frame()`

### `utils.py`

| Function | Description |
|---|---|
| `save_object(file_path, obj)` | Pickles any Python object to disk |
| `load_object(file_path)` | Unpickles an object from disk |
| `evaluate_models(X_train, y_train, X_test, y_test, models, params)` | Runs GridSearchCV per model, applies best params, returns R² scores dict |

---

## Models Trained

| # | Model | Library |
|---|---|---|
| 1 | Linear Regression | scikit-learn |
| 2 | Decision Tree Regressor | scikit-learn |
| 3 | Random Forest Regressor | scikit-learn |
| 4 | Gradient Boosting Regressor | scikit-learn |
| 5 | AdaBoost Regressor | scikit-learn |
| 6 | K-Nearest Neighbors Regressor | scikit-learn |
| 7 | XGBoost Regressor | xgboost |
| 8 | CatBoost Regressor | catboost |
| 9 | Logistic Regression | scikit-learn |

---

## Hyperparameter Tuning

`GridSearchCV` with `cv=3` and `n_jobs=-1` is applied to all models with tunable parameters.

| Model | Parameters Tuned |
|---|---|
| Random Forest | `n_estimators`: [50,100,200] · `max_depth`: [None,5,10] · `min_samples_split`: [2,5] |
| Gradient Boosting | `n_estimators`: [50,100,200] · `learning_rate`: [0.01,0.05,0.1] · `max_depth`: [3,5,7] |
| AdaBoost | `n_estimators`: [50,100,200] · `learning_rate`: [0.01,0.1,1.0] |
| CatBoost | `iterations`: [100,200] · `learning_rate`: [0.01,0.05,0.1] · `depth`: [4,6,8] |
| XGBoost | `n_estimators`: [50,100,200] · `learning_rate`: [0.01,0.05,0.1] · `max_depth`: [3,5,7] |
| KNN | `n_neighbors`: [3,5,7,9] · `weights`: [uniform,distance] |
| Decision Tree | `max_depth`: [None,5,10,15] · `min_samples_split`: [2,5,10] · `criterion`: [squared_error,friedman_mse] |
| Logistic Regression | `C`: [0.1,1.0,10.0] · `max_iter`: [100,200,500] · `solver`: [lbfgs,saga] |
| Linear Regression | *(no tuning — no meaningful hyperparameters)* |

---

## Web Application

Built with **Flask**, served at `http://localhost:1010`.

### Routes

| Route | Method | Description |
|---|---|---|
| `/` | GET | Landing page (`index.html`) |
| `/predict` | GET | Prediction form (`home.html`) |
| `/predict` | POST | Accepts form input, returns predicted math score |

### Form Input Fields

| Field | Type | Options |
|---|---|---|
| Gender | Dropdown | male, female |
| Race/Ethnicity | Dropdown | group A – group E |
| Parental Level of Education | Dropdown | associate's, bachelor's, master's, high school, some college, some high school |
| Lunch | Dropdown | standard, free/reduced |
| Test Preparation Course | Dropdown | none, completed |
| Writing Score | Number (0–100) | free input |
| Reading Score | Number (0–100) | free input |

### Request Flow

```
User fills form on /predict (home.html)
        │
        ▼  POST /predict
    app.py → predict_datapoint()
        │
        ▼
    CustomData.get_data_as_data_frame()
        │  → 1-row Pandas DataFrame
        ▼
    PredictPipeline.predict(features)
        │  → preprocessor.pkl transforms input
        │  → model.pkl predicts math score
        ▼
    Result displayed on home.html
```

---

## How to Clone & Run

### Prerequisites
- Python 3.8+
- `pip`
- `git`

---

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd "ML Project"
```

---

### 2. Create and activate a virtual environment

```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> `setup.py` registers `src/` as the `mlproject` package so that `from src.utils import ...` style imports work everywhere. For a development (editable) install, uncomment `-e .` in `requirements.txt` and run `pip install -e .`

---

### 4. Train the model

```bash
python src/components/data_ingestion.py
```

Runs the full pipeline: reads data → train/test split → preprocessing → trains & tunes 9 models → saves `artifacts/preprocessor.pkl` and `artifacts/model.pkl`.

Expected console output:
```
(0.87, 'CatBoost Regressor')   # example — actual values may vary
```

---

### 5. Start the Flask app

```bash
python app.py
```

Open **http://localhost:1010/predict**, fill in the form, and click **"Predict your Maths Score"**.

---

## Logging & Exception Handling

### Logging (`src/logger.py`)
- Creates a new timestamped file on every run: `logs/log_MM-DD-YYYY_HH-MM-SS.log`
- Writes simultaneously to **the log file and the terminal** (`FileHandler` + `StreamHandler`)
- Format: `[timestamp] line_number module - LEVEL - message`

### Exception Handling (`src/exception.py`)
- `CustomException` enriches every exception with the **filename** and **line number** of the original failure
- Example error message:
  ```
  Error occured in python script name [src/pipeline/predict_pipeline.py]
  line number [16]
  error message [No such file or directory: 'artifacts/model.pkl']
  ```
- Standard usage pattern across the codebase:
  ```python
  except Exception as e:
      raise CustomException(e, sys)
  ```