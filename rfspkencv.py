import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from scipy.stats import spearmanr, kendalltau

# 1) LOAD & CLEAN
data = pd.read_csv('data_imputed.csv')
if 'status' in data.columns:
    data = data.drop('status', axis=1)
data = data[data['stage'].notna()].reset_index(drop=True)

# 2) SPLIT OFF X & y
X = data.drop('stage', axis=1).copy()
y_raw = data['stage'].copy()

# 3) ENCODE CATEGORICAL FEATURES IN X
for col in X.select_dtypes(include=['object']).columns:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))

# 4) ENCODE TARGET y
y = LabelEncoder().fit_transform(y_raw)

# 5) RF IMPORTANCES ON FULL DATA → top5 & top10
rf_full = RandomForestClassifier(n_estimators=100, random_state=42)
rf_full.fit(X, y)
rf_imp_full = pd.Series(rf_full.feature_importances_, index=X.columns)\
                   .sort_values(ascending=False)
top5_rf_full    = rf_imp_full.index[:5].tolist()
top10_rf_full   = rf_imp_full.index[:10].tolist()

print("RF – top 5 features on FULL data:")
print(top5_rf_full)

# 6) RF IMPORTANCES ON REDUCED (TOP-10) DATA → top5
rf_reduced = RandomForestClassifier(n_estimators=100, random_state=42)
rf_reduced.fit(X[top10_rf_full], y)
rf_imp_reduced = pd.Series(rf_reduced.feature_importances_, index=top10_rf_full)\
                      .sort_values(ascending=False)
top5_rf_reduced = rf_imp_reduced.index[:5].tolist()

print("\nRF – top 5 features on REDUCED (top-10) data:")
print(top5_rf_reduced)

# 7) SPEARMAN ON FULL DATA → top5 & top10
spearman_full = {}
for col in X.columns:
    corr, _ = spearmanr(X[col], y)
    spearman_full[col] = abs(corr)
    
spearman_full = pd.Series(spearman_full).sort_values(ascending=False)
top5_sp_full   = spearman_full.index[:5].tolist()
top10_sp_full  = spearman_full.index[:10].tolist()

print("\nSpearman – top 5 features on FULL data:")
print(top5_sp_full)

# 8) SPEARMAN ON REDUCED (TOP-10) DATA → top5
# Use the top 10 features from Spearman on full data to create a reduced dataset
# Then recalculate Spearman on this reduced set to get top 5
spearman_reduced = {}
for col in top10_sp_full:
    corr, _ = spearmanr(X[col], y)
    spearman_reduced[col] = abs(corr)

spearman_reduced = pd.Series(spearman_reduced).sort_values(ascending=False)
top5_sp_reduced = spearman_reduced.index[:5].tolist()

print("\nSpearman – top 5 features on REDUCED (top-10) data:")
print(top5_sp_reduced)

# 9) KENDALL ON FULL DATA → top5 & top10
kendall_full = {}
for col in X.columns:
    corr, _ = kendalltau(X[col], y)
    kendall_full[col] = abs(corr)
    
kendall_full = pd.Series(kendall_full).sort_values(ascending=False)
top5_kd_full = kendall_full.index[:5].tolist()
top10_kd_full = kendall_full.index[:10].tolist()

print("\nKendall – top 5 features on FULL data:")
print(top5_kd_full)

# 10) KENDALL ON REDUCED (TOP-10) DATA → top5
# Use the top 10 features from Kendall on full data to create a reduced dataset
# Then recalculate Kendall on this reduced set to get top 5
kendall_reduced = {}
for col in top10_kd_full:
    corr, _ = kendalltau(X[col], y)
    kendall_reduced[col] = abs(corr)

kendall_reduced = pd.Series(kendall_reduced).sort_values(ascending=False)
top5_kd_reduced = kendall_reduced.index[:5].tolist()

print("\nKendall – top 5 features on REDUCED (top-10) data:")
print(top5_kd_reduced)

# 11) SHOW THE IMPORTANCES / CORRELATIONS FOR EACH 5-FEATURE SET
print("\n--- Feature importances / correlation values for each 5-feature subset ---")

print("\nRF_full_top5 importances:")
print(rf_imp_full[top5_rf_full])

print("\nRF_reduced_top5 importances:")
print(rf_imp_reduced[top5_rf_reduced])

print("\nSP_full_top5 |Spearman| values:")
print(spearman_full[top5_sp_full])

print("\nSP_reduced_top5 |Spearman| values:")
print(spearman_reduced[top5_sp_reduced])

print("\nKD_full_top5 |Kendall| values:")
print(kendall_full[top5_kd_full])

print("\nKD_reduced_top5 |Kendall| values:")
print(kendall_reduced[top5_kd_reduced])

# 12) 5-FOLD CROSS-VALIDATION ON EACH 5-FEATURE SUBSET
cv = 5
rf = RandomForestClassifier(n_estimators=100, random_state=42)

cases = [
    ("RF_full_top5",    top5_rf_full),
    ("RF_reduced_top5", top5_rf_reduced),
    ("SP_full_top5",    top5_sp_full),
    ("SP_reduced_top5", top5_sp_reduced),
    ("KD_full_top5",    top5_kd_full),
    ("KD_reduced_top5", top5_kd_reduced)
]

print("\n=== 5-Fold CV Accuracy (mean ± std) on each 5-feature subset ===")
for name, feats in cases:
    scores = cross_val_score(rf, X[feats], y, cv=cv, scoring='accuracy')
    print(f"{name:17s}: {scores.mean():.4f} ± {scores.std():.4f}")


