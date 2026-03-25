"""Gemini API integration for natural language → data analysis code generation."""
import os
import json
import re
import google.generativeai as genai

SYSTEM_PROMPT = """You are an expert data analyst AI assistant. Users ask questions about their dataset and you generate executable Python code to answer those questions with pandas and plotly visualizations.

CRITICAL: Respond with ONLY a valid JSON object — no markdown fences, no text before or after the JSON.

Response format:
{
  "explanation": "Clear, concise explanation of what you analyzed and the key findings",
  "pandas_code": "Python code using pandas. Variable 'df' is the input DataFrame. You MUST create 'result_df' as the output DataFrame.",
  "viz_code": "Python plotly code that uses 'result_df' to create a variable called 'fig'. Use null if no chart is needed.",
  "viz_type": "bar | line | pie | scatter | heatmap | table_only | null"
}

PANDAS CODE RULES:
- Input: df (pandas DataFrame with the user's data)
- Output: result_df (a pandas DataFrame — always required)
- Available imports: pandas as pd, numpy as np
- Use actual column names from the schema provided
- Handle missing values with .fillna() or .dropna() where needed
- For year-over-year comparisons: filter df['year'] == X
- For rankings: use .nlargest() or .sort_values().head()
- Keep result_df focused and readable (avoid too many columns)

VIZ CODE RULES:
- DO NOT include any import statements — px, go, pd, np are already available
- Input: result_df (the DataFrame produced by pandas_code)
- Output: fig (a plotly Figure)
- Always set a descriptive title
- Use color parameters for grouped charts
- For grouped bar charts comparing years: use px.bar with barmode='group'
- For time series: use px.line
- Make axis labels human-readable
- Set appropriate figure height (600-700px)

CHART SELECTION GUIDE:
- Comparing values across categories → bar chart
- Year-over-year comparison → grouped bar chart
- Trends over time → line chart
- Parts of a whole → pie chart
- Correlation between two numeric values → scatter chart
- Ranking / top N → horizontal bar chart

EXAMPLE for "top 5 highest paid employees in 2023":
pandas_code: "result_df = df[df['year'] == 2023].nlargest(5, 'salary')[['full_name', 'job_title', 'department', 'salary']].reset_index(drop=True)"
viz_code: "fig = px.bar(result_df, x='full_name', y='salary', color='department', title='Top 5 Highest Paid Employees (2023)', labels={'salary': 'Annual Salary ($)', 'full_name': 'Employee'}, height=500); fig.update_layout(xaxis_tickangle=-30)"
"""


class LLMHandler:
    def __init__(self):
        self.chat = None
        self.conversation_history = []

    def _get_model(self):
        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set.")
        genai.configure(api_key=api_key)
        return genai.GenerativeModel("gemini-2.0-flash", system_instruction=SYSTEM_PROMPT)

    def reset_conversation(self):
        self.conversation_history = []
        self.chat = None

    def generate_analysis(self, question: str, schema_info: str, sample_rows: str) -> dict:
        user_message = (
            "Dataset Schema:\n" + schema_info +
            "\n\nSample data (first 5 rows):\n" + sample_rows +
            "\n\nUser question: " + question +
            "\n\nGenerate the pandas + plotly code to answer this question. Remember: respond with ONLY the JSON object."
        )

        if self.chat is None:
            model = self._get_model()
            self.chat = model.start_chat(history=[])

        response = self.chat.send_message(user_message)
        raw_text = response.text

        return self._parse_response(raw_text)

    def _parse_response(self, raw_text: str) -> dict:
        text = raw_text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

        try:
            result = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{[\s\S]*\}", text)
            if match:
                try:
                    result = json.loads(match.group())
                except json.JSONDecodeError:
                    result = {
                        "explanation": "Could not parse LLM response. Please try rephrasing your question.",
                        "pandas_code": "result_df = df.head(10)",
                        "viz_code": None,
                        "viz_type": "table_only",
                    }
            else:
                result = {
                    "explanation": raw_text,
                    "pandas_code": "result_df = df.head(10)",
                    "viz_code": None,
                    "viz_type": "table_only",
                }

        return {
            "explanation": result.get("explanation", ""),
            "pandas_code": result.get("pandas_code", "result_df = df.head(10)"),
            "viz_code": result.get("viz_code"),
            "viz_type": result.get("viz_type", "table_only"),
        }
