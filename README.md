# Respiratory Admissions: Predicting 30-Day Readmissions from Synthetic NHS HES Data

This project develops an end-to-end machine-learning pipeline to predict 30-day emergency readmissions for patients admitted with respiratory conditions.

Synthetic HES APC-style data is processed into patient-level records, analysed, modelled, and finally deployed in an interactive dashboard.

## Project Objectives
- Convert spell-level administrative data into clean patient-level records.
- Explore demographic and clinical patterns in respiratory admissions.
- Train and compare models to predict 30-day readmission risk.
- Optimise thresholds, assess errors, and evaluate fairness.
- Provide an interactive dashboard to explore predictions and high-risk cohorts.

## Key Findings
- Several simple engineered features (notably number of prior spells, mean LOS, and emergency proportion) carry strong predictive information.
- Logistic regression performs strongly, however XGBoost delivers the best balance of PR-AUC, log-loss, and interpretability when calibrated and threshold-optimised.
- SHAP analysis confirms `n_spells` as the dominant driver of risk, with smaller contributions from `pct_emerg`, `mean_los`, and clinical grouping.
- Performance is stable across train/CV/test, and error analysis indicates plausible failure modes rather than systematic bias.
- Exploratory K-means clustering did not reveal meaningful patient subgroups and is excluded from the final workflow.

## Project Structure
```bash
respiratoryAdmissions/
├─ data/
│  ├─ raw/           # synthetic HES APC CSV files + IMD lookup
│  ├─ processed/     # cleaned spell-level and patient-level data, and IMD
├─ models/
│  ├─ readmission_xgb_pipeline.pkl   # final model pipeline (preprocessing + XGB)
├─ notebooks/
│  ├─ 01_data_quality.ipynb
│  ├─ 02_exploratory_analysis.ipynb
│  ├─ 03_patient_summary.ipynb
│  ├─ 04_modelling.ipynb             # full ML workflow
├─ tools/
│  ├─ ethnicity_map.py
│  ├─ process_imd.py
│  ├─ repair_synthetic_hes.py        # full synthetic HES repair script
├─ app.py                            # Streamlit dashboard
├─ README.md
├─ requirements.txt
```

## Workflow Overview

**1. Data Repair & Cleaning**
- Corrects date inconsistencies and length-of-stay anomalies
- Standardises IMD and ethnicity fields
- Generates reproducible spell-level and patient-level datasets

**2. Exploratory Analysis**
- Distributions of age, LOS, emergency use, deprivation
- Condition-specific patterns
- Light inferential checks to detect meaningful group differences

**3. Patient-Level Feature Engineering**
- Aggregates spells to patients
- Final modelling features include: `age`, `sex`, `ethnicity_group`, `respiratory_group_mode`, `n_spells`, `mean_los`, `pct_emerg`, `imd_quintile`
- Labels readmission within 30 days

**4. Modelling**
- 60/20/20 train–validation–test split with stratification
- Baseline: Logistic Regression
- Final model: XGBoost with early stopping + hyperparameter tuning
- Precision–recall threshold optimisation
- Error inspection (top false positives/negatives)
- SHAP global + local interpretability

**_Exploratory Only: Clustering_**

_K-means was tested (k=2–8). Scores were consistently weak and produced no meaningful cohorts, so clustering is not used in the final approach._

**5. Dashboard**

The Streamlit app is available here:
https://khizer-i-respiratory-admissions-app-dqmkum.streamlit.app/

`app.py` provides a one-page Streamlit dashboard featuring:
- Model performance snapshot: PR curve, F1-vs-threshold, feature importance
- Risk landscape: mean predicted risk by respiratory group & IMD quintile
- High-risk cohort explorer: interactive threshold slider + patient table

## Reproducibility (Windows)
```powershell
# 1. Clone the repository
git clone https://github.com/<your-username>/respiratoryAdmissions.git
cd respiratoryAdmissions

# 2. Create virtual environment and install dependencies
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# 3. Generate clean spell-level data
python -m tools.repair_synthetic_hes `
    --in data/raw/<apc_raw_folder> `
    --imd data/processed/imd_clean.parquet `
    --out data/processed/apc_clean.parquet `
    --n_years 3

# 4. Run notebooks 01–03 to generate patient_level.parquet
# 5. Run 04_modelling.ipynb to train and save the model

# 6. Launch the dashboard
streamlit run app.py
```

## Data Sources
- Artificial HES Admitted Patient Care Full
https://digital.nhs.uk/services/artificial-data
- Indices of Multiple Deprivation (2019)
https://www.gov.uk/government/statistics/english-indices-of-deprivation-2019
