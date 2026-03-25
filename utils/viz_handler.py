"""Safe execution of LLM-generated Plotly visualization code."""
import traceback
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def _strip_imports(code: str) -> str:
    """Remove import statements — px, go, pd, np are already in the namespace."""
    lines = [
        line for line in code.splitlines()
        if not line.strip().startswith(("import ", "from "))
    ]
    return "\n".join(lines)


def execute_viz_code(code: str, result_df: pd.DataFrame):
    """
    Execute plotly visualization code in a restricted namespace.
    Returns (fig, error_message).
    """
    if not code or code.strip().lower() in ("null", "none", ""):
        return None, None

    code = _strip_imports(code)

    namespace = {
        "result_df": result_df.copy(),
        "pd": pd,
        "np": np,
        "px": px,
        "go": go,
        "__builtins__": {
            "len": len,
            "range": range,
            "list": list,
            "dict": dict,
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "print": print,
            "sorted": sorted,
            "zip": zip,
            "enumerate": enumerate,
            "min": min,
            "max": max,
            "sum": sum,
            "abs": abs,
            "round": round,
            "isinstance": isinstance,
        },
    }
    try:
        exec(code, namespace)  # noqa: S102
        fig = namespace.get("fig")
        if fig is None:
            return None, "Visualization code did not create 'fig'. Make sure your code assigns a Plotly figure to 'fig'."
        return fig, None
    except Exception:
        return None, f"Visualization error:\n{traceback.format_exc()}"


def make_fallback_chart(result_df: pd.DataFrame, title: str = "Data Overview"):
    """Generate a simple bar chart when LLM viz code fails."""
    numeric_cols = result_df.select_dtypes(include=[np.number]).columns.tolist()
    text_cols = result_df.select_dtypes(include=["object", "category"]).columns.tolist()

    if numeric_cols and text_cols:
        fig = px.bar(
            result_df,
            x=text_cols[0],
            y=numeric_cols[0],
            title=title,
            height=500,
        )
        return fig
    return None
