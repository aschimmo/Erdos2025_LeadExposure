
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import fbeta_score, precision_score, recall_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import clone

# Load your DataFrame here
# df = pd.read_csv('your_data.csv')  # Uncomment and replace with actual data source
# For demonstration: df = your preloaded DataFrame

X = df[['Latitude', 'Longitude', 'YEARBLT']]
y = df['is_lead']

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), ['Latitude', 'Longitude', 'YEARBLT'])
])

base_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', KNeighborsClassifier(n_neighbors=8, weights='distance'))
])

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
thresholds = np.linspace(0.1, 0.9, 81)
best_thresholds = []
f2_scores = []

plt.figure(figsize=(10, 6))
for i, (train_idx, val_idx) in enumerate(skf.split(X_train_full, y_train_full)):
    X_train_fold, X_val_fold = X_train_full.iloc[train_idx], X_train_full.iloc[val_idx]
    y_train_fold, y_val_fold = y_train_full.iloc[train_idx], y_train_full.iloc[val_idx]

    pipeline = clone(base_pipeline)
    pipeline.fit(X_train_fold, y_train_fold)

    probas = pipeline.predict_proba(X_val_fold)[:, 1]
    fold_f2_scores = [fbeta_score(y_val_fold, probas >= t, beta=2) for t in thresholds]
    best_idx = np.argmax(fold_f2_scores)

    best_thresholds.append(thresholds[best_idx])
    f2_scores.append(fold_f2_scores[best_idx])

    plt.plot(thresholds, fold_f2_scores, label=f'Fold {i+1}')

plt.xlabel("Decision Threshold")
plt.ylabel("F2 Score")
plt.title("F2 Score vs. Threshold Across Folds")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

optimal_threshold = np.mean(best_thresholds)
print(f"\nOptimal average threshold across folds: {optimal_threshold:.3f}")
print(f"Mean F2 score across folds: {np.mean(f2_scores):.3f}")

final_pipeline = clone(base_pipeline)
final_pipeline.fit(X_train_full, y_train_full)
test_probas = final_pipeline.predict_proba(X_test)[:, 1]
y_pred = (test_probas >= optimal_threshold).astype(int)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)

print("\n--- Final Evaluation on Holdout Test Set ---")
print(f"Accuracy : {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall   : {recall:.3f}")
print(f"F2 Score : {f2:.3f}")

# Classification outcome map
lat = X_test['Latitude'].values
lon = X_test['Longitude'].values
outcome_color = []
for true, pred in zip(y_test, y_pred):
    if true == 1 and pred == 1:
        outcome_color.append('green')
    elif true == 1 and pred == 0:
        outcome_color.append('red')
    elif true == 0 and pred == 1:
        outcome_color.append('orange')
    else:
        outcome_color.append('gray')

plt.figure(figsize=(10, 8))
plt.scatter(lon, lat, c=outcome_color, s=20, alpha=0.8)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Classification Outcomes on Test Set")
legend_patches = [
    plt.Line2D([0], [0], marker='o', color='w', label='True Positive', markerfacecolor='green', markersize=10),
    plt.Line2D([0], [0], marker='o', color='w', label='False Negative', markerfacecolor='red', markersize=10),
    plt.Line2D([0], [0], marker='o', color='w', label='False Positive', markerfacecolor='orange', markersize=10),
    plt.Line2D([0], [0], marker='o', color='w', label='True Negative', markerfacecolor='gray', markersize=10),
]
plt.legend(handles=legend_patches, loc='best')
plt.grid(True)
plt.tight_layout()
plt.show()

# Confidence plots
positive_mask = y_pred == 1
negative_mask = y_pred == 0

fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)

sc1 = axs[0].scatter(
    lon[positive_mask], lat[positive_mask],
    c=test_probas[positive_mask], cmap='viridis', vmin=0, vmax=1, s=25
)
axs[0].set_title('Predicted Positive (Detected Lead)')
axs[0].set_xlabel("Longitude")
axs[0].set_ylabel("Latitude")
cbar1 = plt.colorbar(sc1, ax=axs[0])
cbar1.set_label("Confidence in Lead Presence (P)")

sc2 = axs[1].scatter(
    lon[negative_mask], lat[negative_mask],
    c=1 - test_probas[negative_mask], cmap='viridis', vmin=0, vmax=1, s=25
)
axs[1].set_title('Predicted Negative (No Lead)')
axs[1].set_xlabel("Longitude")
cbar2 = plt.colorbar(sc2, ax=axs[1])
cbar2.set_label("Confidence in Lead Absence (1 - P)")

plt.suptitle("Test Set Prediction Confidence by Location", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
