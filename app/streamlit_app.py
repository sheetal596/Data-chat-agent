"""
Chat With Your Data - Streamlit app (Telco Customer Churn)

Run with: streamlit run app/streamlit_app.py
"""

import sys
import os

import streamlit as st
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from agent import run_query, load_model_and_metadata

SAMPLE_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "telco_churn.csv")

st.set_page_config(page_title="Chat With Your Data", page_icon="💬", layout="wide")

st.title("💬 Chat With Your Data")
st.write(
    "Ask questions in plain English about a real customer churn dataset. "
    "An LLM writes pandas code on the fly to answer your question, "
    "compute statistics, generate charts, or run predictions with a "
    "trained churn model."
)

model, model_metadata = load_model_and_metadata()

with st.sidebar:
    st.header("Setup")
    api_key_present = bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY"))

    ollama_available = False
    if not api_key_present:
        try:
            import requests
            ollama_available = requests.get("http://localhost:11434/api/tags", timeout=1).status_code == 200
        except Exception:
            ollama_available = False

    if api_key_present:
        st.success("API key detected")
    elif ollama_available:
        st.success("Ollama detected (running locally)")
    else:
        st.warning(
            "No LLM available.\n\n"
            "Set ANTHROPIC_API_KEY or OPENAI_API_KEY, or install Ollama "
            "(ollama.com) and run: ollama pull llama3.1:8b"
        )

    st.divider()

    if model_metadata:
        m = model_metadata["metrics"][model_metadata["best_model"]]
        st.success(f"Model loaded: {model_metadata['best_model']}")
        with st.expander("Model details"):
            st.write(model_metadata["target_description"])
            st.write(f"ROC-AUC: {m['test_roc_auc']}")
            st.write(f"F1 score: {m['test_f1']}")
            st.write(f"Precision: {m['test_precision']} | Recall: {m['test_recall']}")
            st.write(f"Churn rate in data: {model_metadata['churn_rate']:.1%}")
            st.write("Top feature importances:")
            st.json(model_metadata["feature_importance_top15"])
    else:
        st.warning("No trained model found. Run `python src/train_churn_model.py` first.")

    st.divider()
    uploaded_file = st.file_uploader("Or upload your own CSV", type=["csv"])
    use_sample = st.checkbox("Use Telco Customer Churn dataset", value=not uploaded_file)


if uploaded_file is not None and not use_sample:
    df = pd.read_csv(uploaded_file)
    active_model, active_metadata = None, None
elif os.path.exists(SAMPLE_DATA_PATH):
    df = pd.read_csv(SAMPLE_DATA_PATH)
    active_model, active_metadata = model, model_metadata
else:
    st.error("No data available. Please upload a CSV.")
    st.stop()

st.subheader("Data preview")
st.dataframe(df.head(10), use_container_width=True)
st.caption(f"{df.shape[0]} rows x {df.shape[1]} columns")

if active_metadata:
    st.info(
        f"**About this dataset**: real Telco Customer Churn dataset — "
        f"{df.shape[0]} customers, demographics, account details, and "
        f"subscribed services. {active_metadata['target_description']} "
        f"Overall churn rate: {active_metadata['churn_rate']:.1%}."
    )

st.divider()

if "history" not in st.session_state:
    st.session_state.history = []

st.subheader("Ask a question")

if active_metadata:
    example_questions = [
        "What is the churn rate by contract type?",
        "Show a histogram of monthly charges",
        "What is the average tenure for churned vs non-churned customers?",
        "Predict churn for a customer with tenure=2, Contract='Month-to-month', MonthlyCharges=85",
    ]
else:
    example_questions = [
        "What are the column names and types?",
        "Show summary statistics for all numeric columns",
        "Show a histogram of the first numeric column",
        "Show the top 5 rows sorted by the first numeric column",
    ]

cols = st.columns(len(example_questions))
for i, q in enumerate(example_questions):
    if cols[i].button(q, use_container_width=True):
        st.session_state.pending_question = q

question = st.text_input(
    "Your question",
    value=st.session_state.pop("pending_question", "") if "pending_question" in st.session_state else "",
    placeholder="e.g. What is the churn rate by contract type?",
)

if st.button("Ask", type="primary") and question:
    if not (api_key_present or ollama_available):
        st.error("Please set ANTHROPIC_API_KEY/OPENAI_API_KEY, or install and run Ollama.")
    else:
        with st.spinner("Thinking..."):
            result, code, fig, error = run_query(question, df, active_model, active_metadata)
        st.session_state.history.insert(0, {
            "question": question,
            "result": result,
            "code": code,
            "fig": fig,
            "error": error,
        })

for entry in st.session_state.history:
    st.markdown(f"**Q: {entry['question']}**")

    if entry["error"]:
        st.error("The generated code raised an error:")
        st.code(entry["error"], language="text")
    else:
        if entry["fig"] is not None:
            st.pyplot(entry["fig"])
        elif isinstance(entry["result"], (pd.DataFrame, pd.Series)):
            st.dataframe(entry["result"], use_container_width=True)
        else:
            st.write(entry["result"])

    with st.expander("Show generated code"):
        st.code(entry["code"], language="python")

    st.divider()
