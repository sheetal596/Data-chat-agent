# Chat With Your Data

An LLM-powered data analysis and prediction agent. Ask questions in plain English about a real customer churn dataset. The app writes and executes Python code on the fly to answer your question, compute statistics, generate charts, or run predictions with a trained machine learning model.

## Dataset and Model

This project uses the **Telco Customer Churn dataset** (7,043 customers, 21 columns): demographics, account information, and subscribed services, with a binary label for whether the customer churned.

A **Random Forest** classifier is trained on this data inside a full scikit-learn preprocessing pipeline. Model performance and feature importance are saved to `models/churn_model_metadata.json`.

**Model performance:**
- Cross-validated ROC-AUC: ~0.85
- Test F1 score: ~0.63 on the minority (churn) class
- Overall churn rate in dataset: ~26.5%

## How It Works

The app handles two types of questions:

**Analysis questions** ("What is the churn rate by contract type?")
The LLM generates pandas/matplotlib code, which runs in a sandboxed environment and returns a table, number, or chart.

**Prediction questions** ("Predict churn for a customer with tenure=2, Contract=Month-to-month, MonthlyCharges=85")
The LLM extracts feature values from your question and passes them through the trained model pipeline to return a real churn probability.

The generated code is always shown alongside the result so you can see exactly what ran.

## Project Structure

```
data-chat-agent/
├── app/
│   └── streamlit_app.py
├── data/
│   └── telco_churn.csv
├── models/
│   ├── churn_model.joblib
│   └── churn_model_metadata.json
├── src/
│   ├── agent.py
│   ├── train_churn_model.py
│   └── test_agent.py
├── requirements.txt
└── README.md
```

## Getting Started

**1. Install dependencies**

```bash
pip install -r requirements.txt
```

**2. Train the model**

```bash
cd src
python train_churn_model.py
```

**3. Set up an LLM**

Option A — Anthropic Claude (paid, small free credit on signup):

```bash
export ANTHROPIC_API_KEY=your_key_here
```

Option B — OpenAI GPT (paid, small free credit on signup):

```bash
export OPENAI_API_KEY=your_key_here
```

Option C — Ollama (free, runs locally, no API key needed):

```bash
ollama pull llama3.1:8b
ollama serve
```

The app auto-detects Ollama if no API key is set.

**4. Run the app**

```bash
streamlit run app/streamlit_app.py
```

**5. Run the tests (no API key needed)**

```bash
cd src
python test_agent.py
```

## Example Questions

- What is the churn rate by contract type?
- Show a histogram of monthly charges
- What is the average tenure for churned vs non-churned customers?
- Which payment method has the highest churn rate?
- Predict churn for a customer with tenure=2, Contract=Month-to-month, MonthlyCharges=85

## Key Findings

- Customers on month-to-month contracts churn at ~43%, vs ~11% for one-year and ~3% for two-year contracts
- Tenure and contract type are the strongest predictors of churn

## Tech Stack

- Python
- pandas and NumPy for data manipulation
- scikit-learn for preprocessing pipeline, model training, and cross-validation
- matplotlib for chart generation
- Streamlit for the interactive web app
- Anthropic Claude or OpenAI GPT or Ollama for natural language to code generation

## Future Improvements

- Add SHAP-based explanations for individual predictions
- Add conversation memory so follow-up questions reference prior results
- Allow users to upload their own dataset and retrain the model within the app
- Deploy on AWS with S3 for file and model storage

## Dataset Source

Telco Customer Churn dataset, originally published by IBM, available on Kaggle under CC0 license:
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

## License

MIT