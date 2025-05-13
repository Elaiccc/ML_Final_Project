# Remember to run "pip install scikit-learn" in terminal before using

import pickle
import re

# Load saved model and vectorizer
with open("logistic_model.pkl", "rb") as f:
    clf = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

# Clean input text
def clean_text(s):
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"http\S+", "", s)
    s = re.sub(r"[^a-z0-9,.!? ]+", "", s)
    return s.strip()

# Predict function
def predict_fake_news(article_text):
    cleaned = clean_text(article_text)
    vec = tfidf.transform([cleaned])
    prediction = clf.predict(vec)[0]
    probability = clf.predict_proba(vec)[0]
    return prediction, probability

# Test Example
if __name__ == "__main__":
    user_input = input("📰 Enter a news article:\n")
    pred_label, probs = predict_fake_news(user_input)
    print(f"\n📢 Prediction: This article is likely **{pred_label.upper()}**")
    print(f"Probability → Real: {probs[0]:.2%}, Fake: {probs[1]:.2%}")
