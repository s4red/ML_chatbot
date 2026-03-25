"""
AI Data Chatbot — Ask questions about your dataset, get tables and charts.
Usage: streamlit run app.py
"""
import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

from utils.data_handler import (
    load_dataframe,
    get_schema_info,
    get_sample_rows,
    execute_pandas_code,
)
from utils.viz_handler import execute_viz_code, make_fallback_chart
from utils.llm_handler import LLMHandler

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Data Chatbot",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .chat-header { font-size: 1.6rem; font-weight: 700; margin-bottom: 0.2rem; }
    .stChatMessage { border-radius: 12px; }
    div[data-testid="stSidebar"] { background-color: #f8f9fa; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []          # {role, content, result_df?, fig?, error?}
if "df" not in st.session_state:
    st.session_state.df = None
if "llm" not in st.session_state:
    st.session_state.llm = LLMHandler()
if "schema_info" not in st.session_state:
    st.session_state.schema_info = ""
if "sample_rows" not in st.session_state:
    st.session_state.sample_rows = ""
if "data_source" not in st.session_state:
    st.session_state.data_source = ""

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 AI Data Chatbot")
    st.markdown("Ask questions about your data in plain English.")
    st.divider()

    # Data source selection
    st.markdown("### Data Source")
    data_option = st.radio(
        "Choose data:",
        ["Use sample dataset (employees)", "Upload your own file"],
        key="data_option",
    )

    if data_option == "Upload your own file":
        uploaded = st.file_uploader(
            "Upload CSV or Excel",
            type=["csv", "xlsx", "xls"],
            help="Max 200MB. Columns with year/date enable time comparisons.",
        )
        if uploaded:
            df, err = load_dataframe(uploaded)
            if err:
                st.error(err)
            elif df is not None:
                st.session_state.df = df
                st.session_state.schema_info = get_schema_info(df)
                st.session_state.sample_rows = get_sample_rows(df)
                st.session_state.data_source = uploaded.name
                st.session_state.messages = []
                st.session_state.llm.reset_conversation()
                st.success(f"Loaded {len(df):,} rows × {len(df.columns)} columns")
    else:
        sample_path = "sample_data/employees.csv"
        if os.path.exists(sample_path):
            df = pd.read_csv(sample_path)
            if st.session_state.data_source != "employees.csv":
                st.session_state.df = df
                st.session_state.schema_info = get_schema_info(df)
                st.session_state.sample_rows = get_sample_rows(df)
                st.session_state.data_source = "employees.csv"
                st.session_state.messages = []
                st.session_state.llm.reset_conversation()
            st.success(f"Loaded {len(df):,} rows × {len(df.columns)} columns")
        else:
            st.error("Sample data not found. Run `python generate_sample_data.py` first.")

    # Dataset preview
    if st.session_state.df is not None:
        st.divider()
        st.markdown("### Dataset Preview")
        with st.expander("Show schema"):
            st.code(st.session_state.schema_info, language="text")
        with st.expander("Show first 5 rows"):
            st.dataframe(st.session_state.df.head(), use_container_width=True)

    # Example queries
    st.divider()
    st.markdown("### Example Queries")
    example_queries = [
        "Compare 2022 vs 2023 highest paid employees by job title — show as chart and table",
        "What is the average salary by department for each year?",
        "Show the top 10 highest paid employees in 2023",
        "Which department has the highest salary growth from 2021 to 2024?",
        "Show salary distribution by location in 2023",
        "How many employees are in each department?",
        "Compare average salaries across job titles in Engineering in 2023",
    ]
    for q in example_queries:
        if st.button(q, key=f"ex_{q[:20]}", use_container_width=True):
            st.session_state.pending_query = q

    # Reset
    st.divider()
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.llm.reset_conversation()
        st.rerun()

# ── Main chat area ─────────────────────────────────────────────────────────────
st.markdown('<div class="chat-header">📊 AI Data Chatbot</div>', unsafe_allow_html=True)
st.markdown("Ask questions about your data — get tables **and** charts automatically.")

if st.session_state.df is None:
    st.info("Load a dataset from the sidebar to get started.")
    st.stop()

# Render existing conversation
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            if msg.get("error"):
                st.error(msg["error"])
            if msg.get("result_df") is not None:
                st.markdown("**Results Table**")
                st.dataframe(msg["result_df"], use_container_width=True)
            if msg.get("fig") is not None:
                st.plotly_chart(msg["fig"], use_container_width=True)


def run_query(user_input: str):
    """Process a user query: call LLM, execute code, display results."""
    if not user_input.strip():
        return

    # Add user message to history and display it
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate analysis via Claude
    with st.chat_message("assistant"):
        with st.spinner("Analyzing your data..."):
            analysis = st.session_state.llm.generate_analysis(
                question=user_input,
                schema_info=st.session_state.schema_info,
                sample_rows=st.session_state.sample_rows,
            )

        explanation = analysis.get("explanation", "")
        pandas_code = analysis.get("pandas_code", "")
        viz_code = analysis.get("viz_code")
        viz_type = analysis.get("viz_type", "table_only")

        # Show explanation
        st.markdown(explanation)

        result_df = None
        fig = None
        error_msg = None

        # Execute pandas code
        if pandas_code:
            result_df, exec_error = execute_pandas_code(pandas_code, st.session_state.df)
            if exec_error:
                error_msg = f"Data processing error: {exec_error}"
                st.error(error_msg)
            elif result_df is not None:
                st.markdown("**Results Table**")
                st.dataframe(result_df, use_container_width=True)

                # Execute viz code
                if viz_code and viz_type not in ("table_only", "null", None):
                    fig, viz_error = execute_viz_code(viz_code, result_df)
                    if viz_error:
                        # Try fallback chart
                        fig = make_fallback_chart(result_df, title=user_input[:60])
                        if fig is None:
                            st.warning(f"Could not render chart: {viz_error}")
                    if fig is not None:
                        st.plotly_chart(fig, use_container_width=True)

        # Store message in history
        assistant_entry = {
            "role": "assistant",
            "content": explanation,
            "result_df": result_df,
            "fig": fig,
            "error": error_msg,
        }
        st.session_state.messages.append(assistant_entry)


# Handle example query button clicks
if "pending_query" in st.session_state:
    pending = st.session_state.pop("pending_query")
    run_query(pending)
    st.rerun()

# Chat input
if user_input := st.chat_input("Ask a question about your data..."):
    run_query(user_input)
    st.rerun()
