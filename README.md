# Chat With Your Data

An LLM-powered data analysis and prediction agent. Ask questions in plain
English about a real customer churn dataset, and the app writes and executes
Python (pandas/matplotlib) code on the fly to answer your question, compute
statistics, generate charts, or run predictions with a trained machine
learning model.

## Dataset and Model

This project uses the **Telco Customer Churn dataset** (7,043 customers, 21
columns): demographics (gender, senior citizen status, dependents), account
information (tenure, contract type, payment method, monthly/total charges),
and which services each customer subscribes to (phone, internet, streaming,
tech support, etc.), with a binary label for whether the customer churned.

A churn classification model (Random Forest, selected via cross-validation
against Logistic Regression) is trained on this data inside a full
preprocessing pipeline (scaling numeric features, one-hot encoding
categorical features). Model performance and feature importance are saved
alongside the model in `models/churn_model_metadata.json`.

**Model performance** (5-fold cross-validated ROC-AUC): **~0.85**, with a
test F1 score of ~0.63 on the minority (churn) class, in line with published
benchmarks for this dataset and reflecting the genuine difficulty of
predicting churn from account data alone. Overall churn rate in the data is
~26.5%.

## How it works

1. You ask a question like "What is the churn rate by contract type?",
   "Show a histogram of monthly charges", or "Predict churn for a customer
   with tenure=2, Contract='Month-to-month', MonthlyCharges=85".
2. The app sends your question, a preview of the data, and information about
   the trained model (including valid categorical values) to an LLM (Claude
   or GPT).
3. For analysis questions, the LLM responds with pandas/matplotlib code,
   executed in a restricted sandbox to produce a table, number, or chart.
4. For prediction questions, the LLM extracts the relevant feature values
   from your question (matching them to valid categories), and the app runs
   them through the trained model's full preprocessing and classification
   pipeline to return a real churn probability.
5. The generated code (or extracted features) is shown alongside the result.

This is a small example of an agentic workflow: the LLM acts as a translator
between natural language and both data analysis code and a trained ML model.

## Project Structure

```
data-chat-agent/
├── app/
│   └── streamlit_app.py        # Streamlit UI
├── data/
│   └── telco_churn.csv          # Real Telco Customer Churn dataset (7,043 rows)
├── models/
│   ├── churn_model.joblib        # Trained classification pipeline
│   └── churn_model_metadata.json # Metrics, feature importance, categorical options
├── src/
│   ├── agent.py                  # Core agent: LLM call + sandboxed execution + prediction
│   ├── train_churn_model.py       # Trains and saves the churn model
│   └── test_agent.py               # Sandbox + model tests
├── requirements.txt
└── README.md
```

## Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model (if not already present)

```bash
cd src
python train_churn_model.py
```

This cleans the data (handling the 11 rows with blank TotalCharges), builds
a preprocessing pipeline (scaling and one-hot encoding), trains Logistic
Regression and Random Forest, selects the better model via 5-fold
cross-validated ROC-AUC, and saves it with metrics and feature importance to
`models/`.

### 3. Set up an LLM

The agent supports three options, checked in this order:

**Option A — Anthropic Claude (paid, small free credit on signup)**
```bash
export ANTHROPIC_API_KEY=your_key_here
```

**Option B — OpenAI GPT (paid, small free credit on signup)**
```bash
export OPENAI_API_KEY=your_key_here
```

**Option C — Ollama (free, runs locally, no API key)**
1. Install from [ollama.com](https://ollama.com)
2. Pull a model: `ollama pull llama3.1:8b`
3. Make sure Ollama is running (`ollama serve`, usually automatic)

No environment variable needed for Ollama — the app detects it automatically
if no API key is set.

### 4. Run the app

```bash
streamlit run app/streamlit_app.py
```

### 5. Run the tests

```bash
cd src
python test_agent.py
```

This verifies the sandbox execution (no API key needed) and that the trained
model loads and produces a real prediction.

## Example Questions

- "What is the churn rate by contract type?"
- "Show a histogram of monthly charges"
- "What is the average tenure for churned vs non-churned customers?"
- "Which payment method has the highest churn rate?"
- "Predict churn for a customer with tenure=2, Contract='Month-to-month', MonthlyCharges=85"

## Key Findings from the Data

- Customers on month-to-month contracts churn at roughly 43%, versus about
  11% for one-year and 3% for two-year contracts, by far the strongest
  predictor of churn.
- Tenure and contract type dominate feature importance, consistent with
  churn being driven primarily by how "locked in" a customer is.

## Safety Notes

The generated analysis code runs via Python's exec() with a restricted set
of builtins (no open, import, eval, __import__, etc.) and only has access to
pandas, matplotlib, and the loaded DataFrame. Prediction requests never
execute arbitrary code: the LLM only extracts feature values, which are
passed through the trained pipeline directly. This is a reasonable sandbox
for a personal project or demo, but should not be treated as a fully secure
sandbox for untrusted multi-user deployments without additional isolation.

## Tech Stack

- Python - core language
- pandas / NumPy - data manipulation
- scikit-learn - preprocessing pipeline, model training, cross-validation
- matplotlib - chart generation
- Streamlit - interactive web app
- Anthropic Claude / OpenAI GPT - natural language to code/feature extraction

## Future Improvements

- Add SHAP-based explanations for individual predictions
- Add conversation memory so follow-up questions can reference prior results
- Allow users to upload their own dataset and retrain the model within the app
- Deploy on AWS (e.g., ECS or Lambda + API Gateway) with S3 for file/model storage
- Stronger sandboxing (subprocess isolation, resource limits, timeouts)

## Dataset Source and License

Telco Customer Churn dataset, originally published by IBM, widely available
on Kaggle under a CC0 (public domain) license:
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

## License

MIT
