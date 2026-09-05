import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

#Loadin data and clean
url = 'https://github.com/YBIFoundation/Dataset/raw/main/Cancer.csv'
cancer = pd.read_csv(url)

# Drop identifier and empty trailing columns if present
cols_to_drop = [col for col in ['id', 'diagnosis', 'Unnamed: 32'] if col in cancer.columns]
X = cancer.drop(columns=cols_to_drop)
y = cancer['diagnosis'].map({'M': 1, 'B': 0})  # 1 = Malignant, 0 = Benign

#Stratified train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=2529, stratify=y
)

#pipeline: standardizes features then trains Logistic Regression
pipeline = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000, random_state=2529)
)
pipeline.fit(X_train, y_train)

#Evaluation
y_pred = pipeline.predict(X_test)

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['Benign', 'Malignant']))