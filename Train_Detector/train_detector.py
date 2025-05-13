import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score

# Update this path to the correct local path of your CSV file
DATA_CSV = "sample_with_synthetic_fakes_15000.csv"
df = pd.read_csv(DATA_CSV)

# Prepare real and fake datasets
df_real = df[["cleaned_text"]].copy().rename(columns={"cleaned_text": "text"})
df_real["label"] = "real"

df_fake = df[["fake_openai"]].copy().rename(columns={"fake_openai": "text"})
df_fake["label"] = "fake"

# Combine and shuffle
df_combined = pd.concat([df_real, df_fake], ignore_index=True)
df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df_combined["text"], df_combined["label"],
    test_size=0.2,
    random_state=42,
    stratify=df_combined["label"]
)

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)

# Train Logistic Regression Model
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_vec, y_train)

# Predictions and Evaluation
y_pred = clf.predict(X_test_vec)

print("Classification Report:\n")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

y_proba = clf.predict_proba(X_test_vec)[:, 1]

# Convert 'real'/'fake' labels to binary: fake → 1, real → 0
y_test_binary = (y_test == 'fake').astype(int)

# Compute ROC-AUC
roc_auc = roc_auc_score(y_test_binary, y_proba)
print("📈 ROC-AUC Score:", roc_auc)