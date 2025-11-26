# Respiratory Admissions & Health Inequalities

This project analyses synthetic Hospital Episode Statistics (HES) data to explore patterns in respiratory admissions, investigate potential inequalities, and build a supervised model to predict 30-day readmission risk. 

The work follows a full end-to-end data science workflow, including data engineering, data quality assessment, exploratory analysis, patient-level summarisation and predictive modelling.

The dataset is artificial and non-identifiable, designed for demonstration only.

## Project Structure
```bash
respiratoryAdmissions/
├─ data/
│  ├─ raw/           # artificial HES APC CSV files + IMD Excel
│  ├─ processed/     # cleaned spell-level data + patient-level data
├─ notebooks/
│  ├─ 01_data_quality.ipynb
│  ├─ 02_exploratory_analysis.ipynb
│  ├─ 03_patient_summary.ipynb
│  ├─ 04_modelling.ipynb
├─ tools/
│  ├─ ethnicity_map.py
│  ├─ process_imd.py
│  ├─ repair_synthetic_hes.py
├─ README.md
├─ requirements.txt
```

## Workflow Overview
1. **Data Ingestion & Cleaning**
Raw APC episode data is processed into a spell-level dataset using a custom repair script (`tools/repair_synthetic_hes.py`). This includes date correction, episode-to-spell collapsing, IMD joining, ethnicity mapping, respiratory diagnosis classification and LOS validation.
Output:
`data/processed/apc_clean.parquet`

2. Data Quality Checks
The first notebook validates the cleaned dataset: missingness, LOS plausibility, spell construction, demographic coverage, and exclusions. Additional adjustments are documented.

3. Exploratory Analysis
Trends and distributions are explored across age, sex, ethnicity, deprivation and respiratory diagnosis. Inequalities are investigated using descriptive statistics and non-parametric tests. Seasonal patterns, LOS behaviour and emergency admission patterns are visualised.

4. Patient-Level Summaries
Spell-level data is aggregated to one row per patient. Features include number of spells, mean LOS, emergency proportion and a 30-day readmission flag. This forms the modelling dataset.
Output:
`data/processed/patient_level.parquet`

5. Unsupervised Clustering (Exploratory Only)
K-means was tested to identify potential patient cohorts. Across k=2–8, cluster-quality metrics were consistently weak and yielded no clinically meaningful groups. Clustering is therefore excluded from the final workflow.

6. Supervised Modelling


## Installation & Setup (Windows)
```powershell
# 1. Clone the repository
git clone https://github.com/<your-username>/respiratoryAdmissions.git
cd respiratoryAdmissions

# 2. Create virtual environment and install dependencies
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# 3. Run the repair script to generate the clean dataset
python tools/repair_synthetic_hes.py`
--in data/raw/artificial_hes_apc_202302_v1_full`
--imd data/processed/imd_clean.parquet`
--out data/processed/apc_clean.parquet`
```

## Data Sources
- Artificial HES APC Data (v2023-02) — NHS Digital Open Data
https://digital.nhs.uk
- Indices of Multiple Deprivation (2019)
https://www.gov.uk/government/statistics/english-indices-of-deprivation-2019
