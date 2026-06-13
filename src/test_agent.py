

import os
import sys

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__)))
from agent import _SAFE_BUILTINS, _clean_code


def _exec_code(code, df):
    local_env = {"df": df, "pd": pd, "plt": plt, "result": None}
    global_env = {"__builtins__": _SAFE_BUILTINS}
    exec(_clean_code(code), global_env, local_env)
    return local_env["result"]


def test_groupby_mean():
    df = pd.DataFrame({"region": ["A", "A", "B"], "revenue": [10, 20, 30]})
    code = "result = df.groupby('region')['revenue'].mean()"
    result = _exec_code(code, df)
    assert result["A"] == 15
    assert result["B"] == 30
    print("test_groupby_mean passed")


def test_chart_generation():
    df = pd.DataFrame({"product": ["A", "B"], "units": [5, 10]})
    code = (
        "df.plot(kind='bar', x='product', y='units')\n"
        "result = 'chart'"
    )
    result = _exec_code(code, df)
    assert result == "chart"
    assert plt.get_fignums()
    plt.close("all")
    print("test_chart_generation passed")


def test_clean_code_strips_fences():
    raw = "```python\nresult = 1 + 1\n```"
    cleaned = _clean_code(raw)
    assert "```" not in cleaned
    local_env = {"result": None}
    exec(cleaned, {"__builtins__": _SAFE_BUILTINS}, local_env)
    assert local_env["result"] == 2
    print("test_clean_code_strips_fences passed")


def test_restricted_builtins_blocks_open():
    code = "result = open('/etc/passwd').read()"
    try:
        _exec_code(code, pd.DataFrame())
        raise AssertionError("Expected NameError for blocked 'open'")
    except NameError:
        print("test_restricted_builtins_blocks_open passed")


def test_real_model_prediction():
    from agent import load_model_and_metadata, predict_churn

    model, metadata = load_model_and_metadata()
    assert model is not None, "Model not found - run train_churn_model.py first"
    assert metadata is not None

    label, proba = predict_churn(model, metadata, {
        "tenure": 1,
        "Contract": "Month-to-month",
        "MonthlyCharges": 70.0,
    })
    assert label in ("Yes", "No")
    assert 0 <= proba <= 1
    print(f"test_real_model_prediction passed (churn={label}, probability={proba})")


if __name__ == "__main__":
    test_groupby_mean()
    test_chart_generation()
    test_clean_code_strips_fences()
    test_restricted_builtins_blocks_open()
    test_real_model_prediction()
    print("\nAll tests passed.")
