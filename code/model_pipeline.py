"""
Mechanic-Centre Revenue Prediction
Author : Harsh Merchant
Goal   : Train tuned RandomForest & XGBoost, pick best, rank potential sites.
Run    : python code/model_pipeline.py
"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import (
    LabelEncoder, PolynomialFeatures, StandardScaler
)
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, make_scorer

# ------------------------------------------------------------------
# 1) paths & data
# ------------------------------------------------------------------
DATA_DIR   = Path("data")
RESULT_DIR = Path("results")
RESULT_DIR.mkdir(exist_ok=True)

current_df   = pd.read_excel(DATA_DIR/"Current Centres.xlsx")
potential_df = pd.read_excel(DATA_DIR/"Potential Centres.xlsx")
TARGET = "ANNUAL_REVENUE"
current_df.dropna(subset=[TARGET], inplace=True)

# ------------------------------------------------------------------
# 2) outlier removal (IQR)
# ------------------------------------------------------------------
q1, q3 = current_df[TARGET].quantile([0.25, 0.75])
iqr    = q3 - q1
lb, ub = q1 - 1.5 * iqr, q3 + 1.5 * iqr
train  = current_df[current_df[TARGET].between(lb, ub)].copy()

# ------------------------------------------------------------------
# 3) feature prep
# ------------------------------------------------------------------
features = [
    'TYRE_BAYS','MOT_BAYS','SERVICE_BAYS','TOTAL_STAFF','AVG_DAILY_STAFF',
    'AVG_SALARY','HOURS_OPEN_PER_WEEK','AREA_EV_PERC',
    'AREA_POPULATION_DENSITY_PPSKM','ANNUAL_RENT','AREA_AFFLUENCE_GRADE'
]

for df in (train, potential_df):
    df['AREA_AFFLUENCE_GRADE'] = (
        df['AREA_AFFLUENCE_GRADE'].fillna('UNKNOWN')
    )

le = LabelEncoder()
le.fit(
    pd.concat([train['AREA_AFFLUENCE_GRADE'],
               potential_df['AREA_AFFLUENCE_GRADE']])
)
for df in (train, potential_df):
    df['AREA_AFFLUENCE_GRADE'] = le.transform(
        df['AREA_AFFLUENCE_GRADE']
    )

X, y = train[features], train[TARGET]
num_cols = [c for c in features if c != 'AREA_AFFLUENCE_GRADE']
cat_cols = ['AREA_AFFLUENCE_GRADE']

num_pipe = Pipeline([
    ('imp', SimpleImputer(strategy='median')),
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('scaler', StandardScaler())
])
prep = ColumnTransformer([
    ('num', num_pipe, num_cols),
    ('cat', 'passthrough', cat_cols)
])

# ------------------------------------------------------------------
# 4) models & hyper-parameter grids
# ------------------------------------------------------------------
rf = Pipeline([('prep', prep),
              ('model', RandomForestRegressor(random_state=42))])
xgb = Pipeline([('prep', prep),
               ('model', XGBRegressor(objective='reg:squarederror',
                                      random_state=42))])

rf_grid = {
    'model__n_estimators':[100,300,500],
    'model__max_depth':[None,5,10],
    'model__min_samples_split':[2,5],
    'model__min_samples_leaf':[1,2]
}
xgb_grid = {
    'model__learning_rate':[0.01,0.05,0.1],
    'model__max_depth':[3,5,7],
    'model__n_estimators':[100,300,500],
    'model__subsample':[0.6,0.8,1.0]
}

scorer = make_scorer(r2_score)
rf_search  = RandomizedSearchCV(rf,  rf_grid,  n_iter=5,
                                scoring=scorer, cv=5,
                                random_state=42, n_jobs=-1)
xgb_search = RandomizedSearchCV(xgb, xgb_grid, n_iter=5,
                                scoring=scorer, cv=5,
                                random_state=42, n_jobs=-1)

rf_search.fit(X, y)
xgb_search.fit(X, y)

best = max([rf_search, xgb_search], key=lambda s: s.best_score_)
print(f"Chosen model : {best.estimator.named_steps['model'].__class__.__name__} "
      f"(CV R²≈{best.best_score_:.3f})")
print("Train R²      :", round(r2_score(y, best.predict(X)), 3))
# ------------------------------------------------------------------
# 5) predict potential sites
# ------------------------------------------------------------------
potential_df['PREDICTED_ANNUAL_REVENUE'] = best.predict(potential_df[features])
ranked = potential_df[['CENTRE_NO','PREDICTED_ANNUAL_REVENUE']]\
         .sort_values('PREDICTED_ANNUAL_REVENUE', ascending=False)
print("\nTop potential centres:\n", ranked.head(10))

ranked.to_csv(RESULT_DIR / "top_potential_centres.csv", index=False)
