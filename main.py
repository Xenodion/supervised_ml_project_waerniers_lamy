import pandas as pd
import matplotlib.pyplot as plt
import os
import joblib
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from utils import modify_pdays
from sklearn.metrics import roc_curve, roc_auc_score
import webbrowser


# Get project path
cwd = os.getcwd()
print(cwd)

# Download test dataset
bank_full = pd.read_csv(os.path.join(cwd, "data", "bank-additional-full.csv"), sep=";")

bank_full.y = bank_full.y.map({"yes": 1, "no":0})
bank_stable = bank_full.iloc[36224:].reset_index(drop=True)

# chronological train/test split
train_size = 0.8
split = int(len(bank_stable)*train_size)
train_set = bank_stable.iloc[:split].copy()
test_set = bank_stable.iloc[split:].copy()

# Split X and y from train dataset
X_test = test_set.drop(columns=['y'])

# Map target
y_test = test_set.y

# Best trained model
model_id = r"saved_pipelines\Random_Forest_pipeline_1763550779.pkl"

# Load corresponding pipeline
pipeline = joblib.load(os.path.join(cwd, model_id))

# Predict
y_pred = pipeline.predict(X_test)

# Get predicted probabilities for the positive class
y_proba = pipeline.predict_proba(X_test)[:, 1]

# Display metrics, objective is to maximize recall since y distribution is very imbalanced
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 score :", f1_score(y_test, y_pred))

new_row = {
    'Pipeline_file': model_id,
    'Precision': precision_score(y_test, y_pred),
    'Recall': recall_score(y_test, y_pred),
    'F1': f1_score(y_test, y_pred),
    'Accuracy': accuracy_score(y_test, y_pred),
}

# Update logs
logs = pd.concat([pd.read_csv(os.path.join(cwd, 'data', 'test_logs.csv')), pd.DataFrame([new_row])], ignore_index=True)
logs.to_csv(os.path.join(os.getcwd(), 'data', 'test_logs.csv'), index=False)

# Plot Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
# plt.show()

# Extract components
preprocess = pipeline.named_steps['preprocessor']
classifier = pipeline.named_steps['classifier']
feature_names = preprocess.get_feature_names_out()
importances = classifier.feature_importances_

feat_imp = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values(by="importance", ascending=False)

plt.figure(figsize=(10, 8))
plt.barh(feat_imp["feature"], feat_imp["importance"])
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Feature Importance from Random Forest")
plt.gca().invert_yaxis()
plt.tight_layout()
# plt.show()

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)

# Compute AUC
roc_auc = roc_auc_score(y_test, y_proba)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--', lw=2, label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()