# Bank Marketing Campaign - Machine Learning Classification Project

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Authors:** Alexandre Waerniers & Vincent Lamy  
**Institution:** Albert School x Mines Paris PSL

## 📋 Project Overview

This project applies supervised machine learning techniques to predict whether a client will subscribe to a bank term deposit based on direct marketing campaign data from a Portuguese banking institution. The campaigns were conducted via phone calls between May 2008 and November 2010.

### 🎯 Objective

Predict whether a client will subscribe (`yes`/`no`) to a term deposit product (variable `y`) using various client features and campaign interaction data.

## 📊 Dataset

- **Source:** Bank Marketing Dataset (UCI Machine Learning Repository)
- **File:** `data/bank-additional-full.csv`
- **Size:** 41,188 examples with 20 input features
- **Target Variable:** `y` (binary: yes/no)
- **Temporal Ordering:** Data is ordered chronologically (May 2008 - November 2010)

### Features Include:
- **Client Information:** age, job, marital status, education, credit default status
- **Campaign Data:** contact type, month, day of week, duration, number of contacts
- **Previous Campaign:** days since last contact, number of previous contacts, outcome
- **Socioeconomic Context:** employment variation rate, consumer price index, consumer confidence index, Euribor 3-month rate, number of employees

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/Xenodion/supervised_ml_project_waerniers_lamy.git
cd supervised_ml_project_waerniers_lamy
```

2. **Create a virtual environment (recommended):**
```bash
python -m venv env_ml
source env_ml/bin/activate  # On Windows: env_ml\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## 📁 Project Structure

```
supervised_ml_project_waerniers_lamy/
│
├── data/
│   ├── bank-additional-full.csv    # Main dataset
│   ├── dataset_content.md          # Dataset variables context
│   └── test_logs.csv               # Test logs
│   └── train_logs.csv              # Train logs
│
├── documents/
│   ├── git_cheat_sheet.md          # GitHub actions 101
│   └── instructions.md             # Project instructions given by instructor
|
├── saved_pipelines/                           # Trained model pipelines
│   ├── MODEL_pipeline_TIMESTAMP1.pkl          # 1st pipeline
│   ├── ...
│   └── MODEL_pipeline_TIMESTAMPn.pkl          # n-th pipeline
│
├── 1_eda.ipynb                     # Data visualization notebook
├── 2_training.ipynb                # Training notebook
├── 3_testing.ipynb                 # Testing notebook
├── utils.py                        # Utility functions
├── requirements.txt                # Project dependencies
└── README.md                       # Project documentation
```

## 🔧 Key Features & Methodology

### 1. **Exploratory Data Analysis (EDA)**
- Statistical analysis and visualization
- Distribution comparisons using custom functions
- Correlation analysis (Pearson for numerical, Cramér's V for categorical)
- Population Stability Index (PSI) for data drift detection

### 2. **Data Preprocessing Pipeline**
- **StandardScaler:** Applied to numerical features (age, campaign, pdays, previous)
- **OneHotEncoder:** Applied to categorical features (job, marital, education, etc.)
- **Feature Engineering:** Custom transformations for specific features
- **Column Transformer:** Modular preprocessing for different feature types

### 3. **Model Training**
We implemented and compared multiple classification algorithms:
- **Logistic Regression** (with L1/L2 regularization)
- **Random Forest Classifier**
- **Gradient Boosting Classifier**
- **XGBoost**
- **CatBoost**

### 4. **Cross-Validation Strategy**
- **TimeSeriesSplit:** Respects temporal ordering of data
- Prevents data leakage in time-dependent datasets
- Maintains chronological integrity for realistic evaluation

### 5. **Hyperparameter Tuning**
- **GridSearchCV:** Systematic parameter optimization
- Multi-metric evaluation (Precision, Recall, F1-Score, Accuracy)
- Custom parameter grids for each model

### 6. **Experiment Tracking**
- Automated logging of all experiments to `data/logs.csv`
- Pipeline persistence using joblib
- Reproducible results with saved configurations

## 📈 Results

Model performance is evaluated using multiple metrics:
- **Accuracy:** Overall correctness
- **Precision:** Positive predictive value
- **Recall:** Sensitivity to positive class
- **F1-Score:** Harmonic mean of precision and recall

Best performing model: **Random Forest**
- Training
    - F1-Score: 0.6844 ± 0.1510
    - Precision: 0.7847 ± 0.0242
    - Recall: 0.6513 ± 0.0244
    - Accuracy: 0.7785 ± 0.0329

- Test
    - Accuracy : 0.694
    - Precision: 0.667
    - Recall   : 0.776
    - F1 score : 0.717

All training results are logged in `data/train_logs.csv` with complete training metadata.

All testing results are logged in `data/test_logs.csv`.

Best model pipeline : `saved_pipelines\Random_Forest_pipeline_1763550779.pkl`

## 💻 Usage

### Training Models

Open and run the Jupyter notebook:
```bash
jupyter notebook 2_training.ipynb
```

The notebook includes:
1. Data loading and exploration
2. Preprocessing pipeline configuration
3. Model training with cross-validation
4. Results visualization and comparison

### Testing Models

Open and run the testing notebook:
```bash
jupyter notebook 3_testing.ipynb
```

The testing notebook automatically:
1. Loads the test dataset
2. Loads the best trained pipeline
3. Makes predictions on unseen data
4. Displays performance metrics (Accuracy, Precision, Recall, F1-Score)
5. Plots confusion matrix for result visualization

## 🛠️ Custom Utilities

The `utils.py` module provides:
- `psi()`: Population Stability Index calculation
- `compare_distributions()`: Visual distribution comparison between datasets
- `plot_heatmap()`: Enhanced correlation matrix visualization
- `cramers_v()`: Association measure for categorical variables
- `train_ts()`: Complete training pipeline with TimeSeriesSplit and logging

## 📊 Evaluation Metrics

All models are evaluated on:
- **Precision:** How many predicted positives are actually positive
- **Recall:** How many actual positives are correctly identified
- **F1-Score:** Balance between precision and recall
- **Accuracy:** Overall prediction correctness

Metrics are computed per fold and aggregated (mean ± std) for robust evaluation.

## 🤝 Contributing

This is an academic project, but suggestions and improvements are welcome!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

## 📝 Citation Request:
  This dataset is publicly available for research. The details are described in [Moro et al., 2014].

  Please include this citation if you plan to use this database:

```
[Moro et al., 2014] S. Moro, P. Cortez and P. Rita. 
A Data-Driven Approach to Predict the Success of Bank Telemarketing. 
Decision Support Systems, In press, http://dx.doi.org/10.1016/j.dss.2014.03.001

  Available at: [pdf] http://dx.doi.org/10.1016/j.dss.2014.03.001
                [bib] http://www3.dsi.uminho.pt/pcortez/bib/2014-dss.txt
```

## 👥 Authors

- **Alexandre Waerniers** - Albert School x Mines Paris PSL
- **Vincent Lamy** - Albert School x Mines Paris PSL

## 🙏 Acknowledgments

- Albert School Paris for the project framework
- UCI Machine Learning Repository for the dataset
- Scikit-learn community for excellent documentation

## 📧 Contact

For questions or collaboration opportunities, please reach out through GitHub issues.

---

**Note:** This project was completed as part of the Machine Learning curriculum at Albert School Paris in collaboration with Mines Paris PSL.