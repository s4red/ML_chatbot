"""Data loading, schema extraction, and safe code execution."""
import io
import traceback
import numpy as np
import pandas as pd


def load_dataframe(uploaded_file) -> tuple[pd.DataFrame, str]:
    """Load a CSV or Excel file into a DataFrame. Returns (df, error_message)."""
    try:
        filename = uploaded_file.name.lower()
        if filename.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file)
        else:
            return None, f"Unsupported file type: {uploaded_file.name}. Use CSV or Excel."
        return df, None
    except Exception as e:
        return None, f"Failed to load file: {e}"


def get_schema_info(df: pd.DataFrame) -> str:
    """Return a concise schema description for the LLM prompt."""
    lines = [f"Shape: {df.shape[0]} rows × {df.shape[1]} columns", "", "Columns:"]
    for col in df.columns:
        dtype = str(df[col].dtype)
        n_unique = df[col].nunique()
        n_null = df[col].isnull().sum()
        if df[col].dtype in [np.int64, np.float64, "int64", "float64"]:
            stats = f"min={df[col].min():.0f}, max={df[col].max():.0f}, mean={df[col].mean():.0f}"
        elif n_unique <= 20:
            sample_vals = df[col].dropna().unique()[:8].tolist()
            stats = f"values: {sample_vals}"
        else:
            sample_vals = df[col].dropna().unique()[:5].tolist()
            stats = f"sample: {sample_vals}"
        null_info = f", {n_null} nulls" if n_null > 0 else ""
        lines.append(f"  - {col} ({dtype}, {n_unique} unique{null_info}): {stats}")
    return "\n".join(lines)


def get_sample_rows(df: pd.DataFrame, n: int = 5) -> str:
    """Return a string representation of the first n rows."""
    return df.head(n).to_string(index=False)


def _strip_imports(code: str) -> str:
    """Remove import statements — pd and np are already in the namespace."""
    lines = [
        line for line in code.splitlines()
        if not line.strip().startswith(("import ", "from "))
    ]
    return "\n".join(lines)


def execute_pandas_code(code: str, df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """
    Execute pandas code in a restricted namespace.
    Returns (result_df, error_message).
    """
    code = _strip_imports(code)
    namespace = {
        "df": df.copy(),
        "pd": pd,
        "np": np,
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
        result_df = namespace.get("result_df")
        if result_df is None:
            return None, "Code did not create 'result_df'. Make sure your code assigns a DataFrame to 'result_df'."
        if not isinstance(result_df, pd.DataFrame):
            return None, f"'result_df' must be a pandas DataFrame, got {type(result_df).__name__}."
        return result_df, None
    except Exception:
        return None, f"Code execution error:\n{traceback.format_exc()}"
