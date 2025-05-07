import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from scipy.stats import spearmanr

# Load the data
data = pd.read_csv('data_imputed.csv')
print("Original dataset shape:", data.shape)

# Drop the 'status' column
if 'status' in data.columns:
    data = data.drop('status', axis=1)
    print("Dataset shape after dropping 'status':", data.shape)

# Handle empty values in the target column
data_clean = data[data['stage'].notna()]  # Remove rows where target is empty
print("Dataset shape after removing empty target values:", data_clean.shape)

# Prepare data and target
X = data_clean.drop('stage', axis=1)
y = data_clean['stage']

print(f"Number of features: {X.shape[1]}")
print(f"Target classes: {y.unique()}")

# Encode categorical features
categorical_cols = X.select_dtypes(include=['object']).columns
numeric_cols = X.select_dtypes(include=['number']).columns
print(f"Categorical columns: {len(categorical_cols)}")
print(f"Numeric columns: {len(numeric_cols)}")

le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    le_dict[col] = le

# Encode target if it's categorical
le_target = LabelEncoder()
y = le_target.fit_transform(y)
print(f"Target classes after encoding: {le_target.classes_}")

# Split data with 75% train, 25% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

############################
# RANDOM FOREST ANALYSIS  #
############################
print("\n--- Random Forest Feature Importance Analysis ---")
rf_all = RandomForestClassifier(n_estimators=100, random_state=42)
rf_all.fit(X_train, y_train)

# Get feature importances
importances = rf_all.feature_importances_
feature_names = X.columns
feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance = feature_importance.sort_values('Importance', ascending=False)

# Top 5 important features from all features
top5_features_rf = feature_importance['Feature'][:5].tolist()
print("Top 5 feature importances from all features (Random Forest):")
print(feature_importance.head(5))

# Get top 10 features
top10_features_rf = feature_importance['Feature'][:10].tolist()
X_train_top10 = X_train[top10_features_rf]
X_test_top10 = X_test[top10_features_rf]

# Feature importance with top 10 features
rf_top10 = RandomForestClassifier(n_estimators=100, random_state=42)
rf_top10.fit(X_train_top10, y_train)

# Get feature importances for top 10
importances_top10 = rf_top10.feature_importances_
feature_importance_top10 = pd.DataFrame({'Feature': top10_features_rf, 'Importance': importances_top10})
feature_importance_top10 = feature_importance_top10.sort_values('Importance', ascending=False)

# Top 5 important features from top 10
top5_from_top10_rf = feature_importance_top10['Feature'][:5].tolist()
print("\nTop 5 feature importances from top 10 features (Random Forest):")
print(feature_importance_top10.head(5))

# Evaluate model with all features
y_pred_all = rf_all.predict(X_test)
accuracy_all = accuracy_score(y_test, y_pred_all)
print("\nAccuracy with all features:", accuracy_all)
print("Classification Report with all features:")
print(classification_report(y_test, y_pred_all, target_names=le_target.classes_))

# Evaluate model with top 5 features
X_train_top5 = X_train[top5_features_rf]
X_test_top5 = X_test[top5_features_rf]
rf_top5 = RandomForestClassifier(n_estimators=100, random_state=42)
rf_top5.fit(X_train_top5, y_train)
y_pred_top5 = rf_top5.predict(X_test_top5)
accuracy_top5 = accuracy_score(y_test, y_pred_top5)
print("\nAccuracy with top 5 features:", accuracy_top5)
print("Classification Report with top 5 features:")
print(classification_report(y_test, y_pred_top5, target_names=le_target.classes_))

############################
# SPEARMAN ANALYSIS       #
############################
print("\n--- Spearman Correlation Feature Importance Analysis ---")

# For the original dataset, we need to encode ALL categorical variables including the target
data_corr = data_clean.copy()
for col in data_corr.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data_corr[col] = le.fit_transform(data_corr[col])

# Get the encoded target column
target_col = 'stage'
target = data_corr[target_col]

# Calculate Spearman correlation with target for all features
feature_corrs = []
for col in data_corr.columns:
    if col != target_col:
        corr, _ = spearmanr(data_corr[col], target)
        # Take the absolute value since we care about strength, not direction
        feature_corrs.append((col, abs(corr)))

# Sort by correlation strength
feature_corrs.sort(key=lambda x: x[1], reverse=True)

# Create a DataFrame for better display of Spearman results
spearman_all_df = pd.DataFrame(feature_corrs, columns=['Feature', 'Correlation'])
spearman_all_df = spearman_all_df.sort_values('Correlation', ascending=False)

# Display top 5 features by Spearman correlation
print("\nTop 5 feature importances from all features (Spearman):")
print(spearman_all_df.head(5))

# Get top 10 features by Spearman correlation
top10_features_spearman = spearman_all_df['Feature'][:10].tolist()

# Select top 10 features and recalculate correlation within this subset
# We need to recalculate to get proper rankings within the top 10
top10_spearman_corrs = []
for col in top10_features_spearman:
    corr, _ = spearmanr(data_corr[col], target)
    top10_spearman_corrs.append((col, abs(corr)))

# Sort by correlation strength
top10_spearman_corrs.sort(key=lambda x: x[1], reverse=True)

# Create DataFrame for top 10 features
spearman_top10_df = pd.DataFrame(top10_spearman_corrs, columns=['Feature', 'Correlation'])
spearman_top10_df = spearman_top10_df.sort_values('Correlation', ascending=False)

# Display top 5 features from top 10 by Spearman correlation
print("\nTop 5 feature importances from top 10 features (Spearman):")
print(spearman_top10_df.head(5))

# Get top 5 features from Spearman 
top5_features_spearman = spearman_all_df['Feature'][:5].tolist()
top5_from_top10_spearman = spearman_top10_df['Feature'][:5].tolist()

# SPEARMAN ACCURACY EVALUATION: ALL FEATURES
print("\n--- Spearman Accuracy Evaluation ---")

# Create a classifier for all features evaluation
rf_all_spearman = RandomForestClassifier(n_estimators=100, random_state=42)
rf_all_spearman.fit(X_train, y_train)  # Use all features
y_pred_all_spearman = rf_all_spearman.predict(X_test)
accuracy_all_spearman = accuracy_score(y_test, y_pred_all_spearman)

print("\nAccuracy with all features (Spearman):", accuracy_all_spearman)
print("Classification Report with all features (Spearman):")
print(classification_report(y_test, y_pred_all_spearman, target_names=le_target.classes_))

# SPEARMAN ACCURACY EVALUATION: TOP 5 FEATURES
# Evaluate model with top 5 Spearman features
X_train_top5_spearman = X_train[top5_features_spearman]
X_test_top5_spearman = X_test[top5_features_spearman]
rf_top5_spearman = RandomForestClassifier(n_estimators=100, random_state=42)
rf_top5_spearman.fit(X_train_top5_spearman, y_train)
y_pred_top5_spearman = rf_top5_spearman.predict(X_test_top5_spearman)
accuracy_top5_spearman = accuracy_score(y_test, y_pred_top5_spearman)

print("\nAccuracy with top 5 features (Spearman):", accuracy_top5_spearman)
print("Classification Report with top 5 features (Spearman):")
print(classification_report(y_test, y_pred_top5_spearman, target_names=le_target.classes_))

# Print summary of results
print("\n--- SUMMARY ---")
print(f"Original dataset shape: {data.shape}")
print(f"Dataset after cleaning: {data_clean.shape}")
print(f"\nRandom Forest:")
print(f"  Top 5 features: {top5_features_rf}")
print(f"  Accuracy with all features: {accuracy_all:.4f}")
print(f"  Accuracy with top 5 features: {accuracy_top5:.4f}")
print(f"  Accuracy difference: {accuracy_all - accuracy_top5:.4f}")
print(f"\nSpearman:")
print(f"  Top 5 features: {top5_features_spearman}")
print(f"  Accuracy with all features: {accuracy_all_spearman:.4f}")
print(f"  Accuracy with top 5 features: {accuracy_top5_spearman:.4f}")
print(f"  Accuracy difference: {accuracy_all_spearman - accuracy_top5_spearman:.4f}")