# =========================================================
# model_utils.py
# Contains ML model training and machine unlearning logic
# =========================================================

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score


# ---------------------------------------------------------
# Function 1: Train baseline ML model (Before Unlearning)
# ---------------------------------------------------------
def train_baseline_model(df):
    """
    Trains Logistic Regression on full dataset
    Returns trained model, vectorizer, accuracy, f1 score
    """

    X = df['clean_text']
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return model, tfidf, acc, f1


# ---------------------------------------------------------
# Function 2: Apply Machine Unlearning (User-Level)
# ---------------------------------------------------------
def unlearn_user(df, target_user):
    """
    Removes a specific user's data and retrains model
    Returns updated model, vectorizer, accuracy, f1 score
    """

    # Remove target user's data
    df_unlearn = df[df['UserId'] != target_user]

    X = df_unlearn['clean_text']
    y = df_unlearn['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return model, tfidf, acc, f1
