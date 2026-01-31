import streamlit as st
import pandas as pd
from model_utils import train_baseline_model, unlearn_user

st.set_page_config(page_title="Machine Unlearning Demo", layout="centered")

st.title("🧠 User-Level Machine Unlearning Demo")
st.write("Demonstration of selective user forgetting in AI models.")

# =========================================================
# STEP 1: Upload Dataset
# =========================================================
uploaded_file = st.file_uploader("Upload Reviews Dataset (CSV)", type=["csv"])

if uploaded_file is None:
    st.warning("Please upload a dataset to continue.")
    st.stop()

# =========================================================
# STEP 2: Load Dataset
# =========================================================
df = pd.read_csv(uploaded_file)
st.success("Dataset loaded successfully!")
st.dataframe(df.head())

# =========================================================
# STEP 3: Select Required Columns
# =========================================================
df = df[['UserId', 'clean_text', 'label']]
df = df.dropna()

# =========================================================
# STEP 4: Train Baseline Model
# =========================================================
if st.button("Train Baseline Model"):
    model, tfidf, acc_before, f1_before = train_baseline_model(df)

    st.session_state['acc_before'] = acc_before
    st.session_state['f1_before'] = f1_before

    st.success(f"Baseline Accuracy: {acc_before:.4f}")
    st.success(f"Baseline F1 Score: {f1_before:.4f}")

# =========================================================
# STEP 5: User-Level Machine Unlearning
# =========================================================
if 'acc_before' in st.session_state:
    st.subheader("🔴 User-Level Machine Unlearning")

    user_list = df['UserId'].value_counts().index.tolist()
    target_user = st.selectbox("Select User to Forget", user_list)

    if st.button("Forget Selected User"):
        model_u, tfidf_u, acc_after, f1_after = unlearn_user(df, target_user)

        st.error(f"Accuracy AFTER Unlearning: {acc_after:.4f}")
        st.error(f"F1 Score AFTER Unlearning: {f1_after:.4f}")

        st.write("### 📊 Comparison")
        st.write(f"Before Unlearning Accuracy: {st.session_state['acc_before']:.4f}")
        st.write(f"After Unlearning Accuracy: {acc_after:.4f}")

        st.write(f"Before Unlearning F1: {st.session_state['f1_before']:.4f}")
        st.write(f"After Unlearning F1: {f1_after:.4f}")

# =========================================================
# STEP 6: Visualization - Before vs After Forgetting
# =========================================================

st.subheader("📊 Model Performance Comparison")

metrics = ['Accuracy', 'F1 Score']
before_values = [
    st.session_state['acc_before'],
    st.session_state['f1_before']
]
after_values = [
    acc_after,
    f1_after
]

fig, ax = plt.subplots()

bar_width = 0.35
x = range(len(metrics))

ax.bar(x, before_values, width=bar_width, label='Before Forgetting')
ax.bar(
    [i + bar_width for i in x],
    after_values,
    width=bar_width,
    label='After Forgetting'
)

ax.set_xlabel("Metrics")
ax.set_ylabel("Score")
ax.set_title("Before vs After User-Level Machine Unlearning")
ax.set_xticks([i + bar_width / 2 for i in x])
ax.set_xticklabels(metrics)
ax.legend()

st.pyplot(fig)

