import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(page_title="Machine Unlearning Demo", layout="centered")

st.title("🧠 User-Level Machine Unlearning")
st.write(
    "This web application demonstrates how an AI model can forget a specific user's "
    "influence while maintaining overall performance."
)

# =========================================================
# STEP 1: UPLOAD REDUCED DATASET (< 25 MB)
# =========================================================
st.subheader("📂 Upload Reduced Dataset")

uploaded_file = st.file_uploader(
    "Upload reduced Amazon reviews dataset (CSV)",
    type=["csv"]
)

if uploaded_file is None:
    st.warning("Please upload the reduced dataset to continue.")
    st.stop()

df = pd.read_csv(uploaded_file)
st.success("Dataset uploaded successfully!")
st.subheader("Dataset Preview")
st.dataframe(df.head())

# =========================================================
# STEP 2: VALIDATE REQUIRED COLUMNS
# =========================================================
required_cols = {"UserId", "Text", "Score"}
if not required_cols.issubset(set(df.columns)):
    st.error(
        "Dataset must contain the following columns: "
        "UserId, Text, Score"
    )
    st.stop()

# =========================================================
# STEP 3: PREPROCESSING
# =========================================================
st.subheader("🧹 Data Preprocessing")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text

df = df.dropna()
df["clean_text"] = df["Text"].apply(clean_text)

# Convert rating to binary sentiment
df = df[df["Score"] != 3]  # remove neutral
df["label"] = df["Score"].apply(lambda x: 1 if x >= 4 else 0)

st.success("Preprocessing completed!")
st.write(df[["UserId", "clean_text", "label"]].head())

# =========================================================
# STEP 4: TRAIN BASELINE MODEL (BEFORE UNLEARNING)
# =========================================================
st.subheader("🤖 Train Baseline Model")

if st.button("Train Baseline Model"):
    X = df["clean_text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    tfidf = TfidfVectorizer(
        max_features=5000,
        stop_words="english"
    )
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)

    acc_before = accuracy_score(y_test, y_pred)
    f1_before = f1_score(y_test, y_pred)

    st.session_state["acc_before"] = acc_before
    st.session_state["f1_before"] = f1_before

    st.success(f"Baseline Accuracy: {acc_before:.4f}")
    st.success(f"Baseline F1 Score: {f1_before:.4f}")

# =========================================================
# STEP 5: USER-LEVEL MACHINE UNLEARNING
# =========================================================
if "acc_before" in st.session_state:
    st.subheader("🔴 User-Level Machine Unlearning")

    user_list = df["UserId"].value_counts().index.tolist()
    target_user = st.selectbox(
        "Select a user to forget",
        user_list
    )

    if st.button("Forget Selected User"):
        df_unlearn = df[df["UserId"] != target_user]

        X_u = df_unlearn["clean_text"]
        y_u = df_unlearn["label"]

        X_train_u, X_test_u, y_train_u, y_test_u = train_test_split(
            X_u, y_u, test_size=0.2, random_state=42, stratify=y_u
        )

        tfidf_u = TfidfVectorizer(
            max_features=5000,
            stop_words="english"
        )
        X_train_u_vec = tfidf_u.fit_transform(X_train_u)
        X_test_u_vec = tfidf_u.transform(X_test_u)

        model_u = LogisticRegression(max_iter=1000)
        model_u.fit(X_train_u_vec, y_train_u)

        y_pred_u = model_u.predict(X_test_u_vec)

        acc_after = accuracy_score(y_test_u, y_pred_u)
        f1_after = f1_score(y_test_u, y_pred_u)

        st.error(f"Accuracy AFTER Unlearning: {acc_after:.4f}")
        st.error(f"F1 Score AFTER Unlearning: {f1_after:.4f}")

        # =================================================
        # STEP 6: VISUALIZATION (BEFORE vs AFTER)
        # =================================================
        st.subheader("📊 Before vs After Comparison")

        metrics = ["Accuracy", "F1 Score"]
        before = [
            st.session_state["acc_before"],
            st.session_state["f1_before"]
        ]
        after = [acc_after, f1_after]

        fig, ax = plt.subplots()
        x = np.arange(len(metrics))
        width = 0.35

        ax.bar(x - width/2, before, width, label="Before Forgetting")
        ax.bar(x + width/2, after, width, label="After Forgetting")

        ax.set_ylabel("Score")
        ax.set_title("Effect of User-Level Machine Unlearning")
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()

        st.pyplot(fig)
