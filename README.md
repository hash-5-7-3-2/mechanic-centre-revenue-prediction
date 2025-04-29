# Mechanic-Centre Revenue Prediction

> Personal data-science project: predicting annual revenue for potential mechanic-centre locations using ensemble learning (Random Forest vs XGBoost).

## Quick Start

```bash
git clone https://github.com/<your-user>/mechanic-centre-revenue-prediction.git
cd mechanic-centre-revenue-prediction
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

"Note: data files are excluded from this repository. Please place Current Centres.csv and Potential Centres.csv in the data/ folder before running."
# place data:
#   data/Current Centres.csv
#   data/Potential Centres.csv
python code/model_pipeline.py
