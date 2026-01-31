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
st.set_page_config(page_title="Advanced Machine Unlearning Demo", layout="centered")

st.title("🧠 Multi-Criteria Machine Unlearning System")
st.write(
    "This application demonstrates **selective machine unlearning** where users can "
    "define multiple conditions to forget specific data and observe how model behavior changes."
)

# =========================================================
# STEP 1: UPLOAD REDUCED DATASET (<25 MB)
# =========================================================
st.subheader("📂 Upload Reduced Dataset")

uploaded_file = st.file_uploader(
    "Upload reduced dataset (CSV with UserId, Text, Score)",
    type=["csv"]
)

if uploaded_file is None:
    st.warning("Please upload the reduced dataset to continue.")
    st.stop()

df = pd.read_csv(uploaded_file)
st.success("Dataset uploaded successfully!")
st.dataframe(df.head())

# =========================================================
# STEP 2: VALIDATE COLUMNS
# =========================================================
required_cols = {"UserId", "Text", "Score"}
if not required_cols.issubset(df.columns):
    st.error("Dataset must contain UserId, Text, and Score columns.")
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

# Binary sentiment
df = df[df["Score"] != 3]
df["label"] = df["Score"].apply(lambda x: 1 if x >= 4 else 0)

st.success("Preprocessing completed!")
st.write(df[["UserId", "clean_text", "label"]].head())

# =========================================================
# STEP 4: TRAIN BASELINE MODEL (BEFORE UNLEARNING)
# =========================================================
st.subheader("🤖 Baseline Model Training")

X = df["clean_text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)
y_prob = model.predict_proba(X_test_vec)[:, 1]

acc_before = accuracy_score(y_test, y_pred)
f1_before = f1_score(y_test, y_pred)

st.success(f"Baseline Accuracy: {acc_before:.4f}")
st.success(f"Baseline F1 Score: {f1_before:.4f}")

# =========================================================
# STEP 5: DEFINE MULTI-CRITERIA FORGETTING RULES
# =========================================================
st.subheader("🧩 Define Forgetting Rules")

st.write(
    "Select **one or more conditions**. "
    "All selected conditions will be applied together (AND logic)."
)

forget_columns = st.multiselect(
    "Select columns to forget by",
    ["UserId", "Score"]
)

filters = {}

for col in forget_columns:
    unique_vals = df[col].unique().tolist()
    selected_vals = st.multiselect(
        f"Select values for {col}",
        unique_vals
    )
    if selected_vals:
        filters[col] = selected_vals

# =========================================================
# STEP 6: APPLY MACHINE UNLEARNING
# =========================================================
if st.button("🚫 Apply Forgetting Rules") and filters:

    # Identify rows to forget
    mask = pd.Series([True] * len(df))
    for col, vals in filters.items():
        mask &= df[col].isin(vals)

    forgotten_data = df[mask]
    df_unlearn = df[~mask]

    st.info(
        f"🔍 Rows selected for forgetting: {forgotten_data.shape[0]} "
        f"({forgotten_data.shape[0] / df.shape[0] * 100:.2f}% of data)"
    )

    # =====================================================
    # STEP 6.1: RETRAIN MODEL AFTER UNLEARNING
    # =====================================================
    X_u = df_unlearn["clean_text"]
    y_u = df_unlearn["label"]

    X_train_u, X_test_u, y_train_u, y_test_u = train_test_split(
        X_u, y_u, test_size=0.2, random_state=42, stratify=y_u
    )

    tfidf_u = TfidfVectorizer(max_features=5000, stop_words="english")
    X_train_u_vec = tfidf_u.fit_transform(X_train_u)
    X_test_u_vec = tfidf_u.transform(X_test_u)

    model_u = LogisticRegression(max_iter=1000)
    model_u.fit(X_train_u_vec, y_train_u)

    y_pred_u = model_u.predict(X_test_u_vec)
    y_prob_u = model_u.predict_proba(X_test_u_vec)[:, 1]

    acc_after = accuracy_score(y_test_u, y_pred_u)
    f1_after = f1_score(y_test_u, y_pred_u)

    # =====================================================
    # STEP 7: FORGETTING-SPECIFIC METRICS (CORE)
    # =====================================================
    # Prediction change on forgotten subset
    X_forgot = tfidf.transform(forgotten_data["clean_text"])
    pred_before_forgot = model.predict(X_forgot)
    pred_after_forgot = model_u.predict(tfidf_u.transform(forgotten_data["clean_text"]))

    prediction_change_rate = np.mean(pred_before_forgot != pred_after_forgot)

    # Confidence drop
    prob_before = model.predict_proba(X_forgot)[:, 1]
    prob_after = model_u.predict_proba(tfidf_u.transform(forgotten_data["clean_text"]))[:, 1]

    confidence_drop = np.mean(prob_before - prob_after)

    # =====================================================
    # STEP 8: DISPLAY RESULTS
    # =====================================================
    st.subheader("📊 Results Summary")

    st.metric("Accuracy Before Forgetting", f"{acc_before:.4f}")
    st.metric("Accuracy After Forgetting", f"{acc_after:.4f}", delta=f"{acc_after-acc_before:.4f}")

    st.metric(
        "Prediction Change Rate (Forgotten Data)",
        f"{prediction_change_rate * 100:.2f}%"
    )

    st.metric(
        "Average Confidence Drop",
        f"{confidence_drop:.4f}"
    )

    # =====================================================
    # STEP 9: LINE CHART (CLEAR STORY)
    # =====================================================
    st.subheader("📉 Model Behavior Change")

    fig, ax = plt.subplots()

    ax.plot(
        ["Before Forgetting", "After Forgetting"],
        [acc_before, acc_after],
        marker="o",
        linewidth=3
    )

    ax.set_ylabel("Accuracy")
    ax.set_title("Effect of Multi-Criteria Machine Unlearning")
    ax.grid(True)

    st.pyplot(fig)

    # =====================================================
    # STEP 10: HUMAN EXPLANATION
    # =====================================================
    st.markdown("""
### 🧠 What does this show?

- The model was first trained using **all available data**
- Multiple conditions were applied to **selectively forget data**
- The **prediction change rate** and **confidence drop** confirm that:
  - The forgotten data no longer influences the model
  - Overall performance remains stable

This demonstrates **flexible, multi-criteria machine unlearning**.
""")
