# 🫁 Respiratory Admissions & Health Inequalities

Exploring hospital respiratory admissions, deprivation, and inequality patterns using synthetic HES data.

## Project Overview
This project analyses hospital admissions for respiratory conditions (COPD, asthma, pneumonia, and related diseases) using synthetic Hospital Episode Statistics (HES APC) data published by NHS Digital.

The goal is to demonstrate a full data analytics and modelling workflow, from raw data cleaning to predictive modelling and visualisation, while examining health inequalities by deprivation, ethnicity, age, and gender.

> ⚠️ **Disclaimer**: All data used are artificial (synthetic) and contain no real patient information. Findings are illustrative and intended purely for learning and portfolio demonstration.

## Project Structure
```bash
respiratoryAdmissions/
├─ data/
│  ├─ raw/           # artificial HES CSV files, IMD Excel
│  ├─ processed/     # cleaned Parquet files
├─ notebooks/        # Jupyter notebooks for analysis
├─ tools/            # repair scripts, ethnicity map, utilities
├─ README.md
├─ requirements.txt
```

## Data Processing Pipeline
1. Data ingestion & repair (`tools/repair_synthetic_hes.py`)
    - Combines yearly HES CSVs
    - Cleans invalid dates, infant records (7001–7007), and sex codes
    - Removes Welsh LSOAs (IMD coverage is England only)
    - Joins IMD and creates quintiles and maps ethnicity codes
    - Builds quinary age bands (0–4, 5–9, …, 85–89, 90+)
    - Filters for respiratory diagnoses (J00–J99)
    - Makes each row its own spell

2. **Output:**
`data/processed/apc_clean.parquet` – spell-level, analysis-ready dataset

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

## About
Developed as a portfolio project to showcase:
- Robust data engineering and cleaning
- Exploratory analysis with an inequalities focus
- Modelling and interpretation using real-world health structures
- Clear, reproducible communication
