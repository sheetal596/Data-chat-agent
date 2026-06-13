

import os
import io
import json
import contextlib
import traceback

import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def load_model_and_metadata():
    """Load the trained churn model and its metadata, if available."""
    model_path = os.path.join(MODEL_DIR, "churn_model.joblib")
    metadata_path = os.path.join(MODEL_DIR, "churn_model_metadata.json")

    if not (os.path.exists(model_path) and os.path.exists(metadata_path)):
        return None, None

    model = joblib.load(model_path)
    with open(metadata_path) as f:
        metadata = json.load(f)

    return model, metadata


def predict_churn(model, metadata, feature_values: dict):
    """
    Run a churn prediction given a dict of feature_name -> value.
    Missing features default to a sensible baseline (median/mode-like values
    are not computed here for simplicity; instead common defaults are used).
    Returns (prediction_label, churn_probability).
    """
    raw_features = metadata["raw_feature_names"]
    numeric_features = set(metadata["numeric_features"])
    categorical_options = metadata["categorical_options"]

    row = {}
    for name in raw_features:
        if name in feature_values:
            row[name] = feature_values[name]
        elif name in numeric_features:
            row[name] = 0
        else:
            # default to the first available category
            row[name] = categorical_options.get(name, ["No"])[0]

    df_row = pd.DataFrame([row])
    proba = model.predict_proba(df_row)[0][1]
    label = "Yes" if proba >= 0.5 else "No"
    return label, round(float(proba), 4)


SYSTEM_PROMPT = """You are a data analysis assistant. You are given a pandas \
DataFrame called `df` and a question about it. Your job is to write Python \
code that answers the question.

Rules:
- The DataFrame is already loaded as `df`. Do not re-create or re-load it.
- Assign your final answer to a variable called `result`.
- `result` can be a pandas DataFrame, Series, scalar (number/string), or None.
- If the question asks for a chart/plot, create it using matplotlib (`plt`),
  which is already imported. Do not call plt.show(). Set `result = "chart"`
  after creating the figure.
- Only use pandas (`pd`), matplotlib (`plt`), and built-in Python. No other
  imports, no file I/O, no network access.
- Return ONLY the Python code, no explanations, no markdown fences.

A trained churn prediction model is also available. If the question asks to \
predict whether a customer will churn, set `result = "PREDICT"` and set a \
variable `predict_features` to a dict mapping feature names to values based \
on the question. Use the exact feature names and, for categorical features, \
one of the exact option values listed below. Omit any feature not mentioned \
in the question (sensible defaults will be used).

{model_context}

Here is a preview of the DataFrame:
{df_info}
"""


def _get_df_info(df: pd.DataFrame) -> str:
    buffer = io.StringIO()
    df.head(5).to_string(buf=buffer)
    preview = buffer.getvalue()
    return f"Columns and dtypes:\n{df.dtypes.to_string()}\n\nFirst 5 rows:\n{preview}"


def generate_code(question: str, df: pd.DataFrame, model_metadata: dict = None) -> str:
    """Call an LLM to generate pandas code answering the question."""
    df_info = _get_df_info(df)

    if model_metadata:
        cat_options = "\n".join(
            f"  - {col}: {opts}" for col, opts in model_metadata["categorical_options"].items()
        )
        model_context = (
            f"Target: {model_metadata['target_description']}\n"
            f"Numeric features: {model_metadata['numeric_features']}\n"
            f"Categorical features and their valid options:\n{cat_options}"
        )
    else:
        model_context = "No trained model is available for this dataset."

    system_prompt = SYSTEM_PROMPT.format(df_info=df_info, model_context=model_context)

    if os.environ.get("ANTHROPIC_API_KEY"):
        return _generate_with_anthropic(system_prompt, question)
    elif os.environ.get("OPENAI_API_KEY"):
        return _generate_with_openai(system_prompt, question)
    elif _ollama_available():
        return _generate_with_ollama(system_prompt, question)
    else:
        raise RuntimeError(
            "No API key found and Ollama is not running. "
            "Set ANTHROPIC_API_KEY/OPENAI_API_KEY, or install Ollama "
            "(ollama.com) and run 'ollama pull llama3.1:8b'."
        )


def _ollama_available() -> bool:
    import requests
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=1)
        return resp.status_code == 200
    except Exception:
        return False


def _generate_with_ollama(system_prompt: str, question: str, model: str = "llama3.1:8b") -> str:
    import requests

    resp = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            "stream": False,
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"].strip()


def _generate_with_anthropic(system_prompt: str, question: str) -> str:
    import anthropic

    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=500,
        system=system_prompt,
        messages=[{"role": "user", "content": question}],
    )
    return response.content[0].text.strip()


def _generate_with_openai(system_prompt: str, question: str) -> str:
    from openai import OpenAI

    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        max_tokens=500,
    )
    return response.choices[0].message.content.strip()


def _clean_code(code: str) -> str:
    """Strip markdown code fences if the model added them anyway."""
    code = code.strip()
    if code.startswith("```"):
        lines = code.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        code = "\n".join(lines)
    return code.strip()


# A small allowlist of safe builtins for the exec environment
_SAFE_BUILTINS = {
    "len": len, "range": range, "min": min, "max": max, "sum": sum,
    "sorted": sorted, "list": list, "dict": dict, "set": set, "tuple": tuple,
    "str": str, "int": int, "float": float, "bool": bool, "round": round,
    "abs": abs, "enumerate": enumerate, "zip": zip, "print": print,
}


def run_query(question: str, df: pd.DataFrame, model=None, model_metadata=None):
    """
    Generate code for the question, execute it safely, and return:
        (result, generated_code, figure_or_none, error_or_none)
    """
    code = generate_code(question, df, model_metadata)
    code = _clean_code(code)

    local_env = {"df": df, "pd": pd, "plt": plt, "result": None, "predict_features": {}}
    global_env = {"__builtins__": _SAFE_BUILTINS}

    fig = None
    error = None

    try:
        plt.close("all")
        with contextlib.redirect_stdout(io.StringIO()):
            exec(code, global_env, local_env)
        result = local_env.get("result")

        if result == "PREDICT":
            if model is None or model_metadata is None:
                result = "No trained model is available."
            else:
                features = local_env.get("predict_features", {})
                label, proba = predict_churn(model, model_metadata, features)
                result = (
                    f"Predicted churn: {label} (probability: {proba:.1%})\n\n"
                    f"Features used: {features}\n"
                    f"(Any unspecified features were filled with defaults.)"
                )
        elif result == "chart" or plt.get_fignums():
            fig = plt.gcf()
    except Exception:
        error = traceback.format_exc()
        result = None

    return result, code, fig, error
