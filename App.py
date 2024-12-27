import streamlit as st
import pandas as pd
import numpy as np
import re
import base64

def clean_numeric_value(value) -> float:
    """Convert a value to a numeric type, handling strings and non-numeric characters."""
    try:
        return float(re.sub(r"[^\d.-]", "", str(value)))
    except (ValueError, TypeError):
        return 0.0

def create_balanced_territories(df, num_territories):
    """Distribute rows into territories with balanced row count and column sums."""
    total_rows = len(df)
    rows_per_territory = total_rows // num_territories
    extra_rows = total_rows % num_territories
    target_sums = df.applymap(clean_numeric_value).sum() / num_territories
    territories = [[] for _ in range(num_territories)]

    # Pre-compute values for efficiency
    df_cleaned = df.applymap(clean_numeric_value)
    df_sorted = df_cleaned.assign(_total=df_cleaned.sum(axis=1)).sort_values('_total', ascending=False).drop(columns='_total')

    # Distribute rows into territories
    for index, row in df_sorted.iterrows():
        row_values = row.values
        best_territory = min(range(num_territories), 
                             key=lambda i: np.sum((np.sum(territories[i], axis=0) + row_values - target_sums) ** 2))
        territories[best_territory].append(row_values)

    # Convert lists of arrays back to DataFrames
    territories = [pd.DataFrame(data, columns=df.columns) for data in territories]

    return territories

def main():
    st.title("Territory Distribution Tool")

    uploaded_file = st.file_uploader("Upload CSV file", type="csv")
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file, encoding='utf-8', sep=None, engine='python')
        except Exception as e:
            st.error(f"Error reading file: {e}")
            return

        st.write("Data Preview:", df.head())
        st.write("Column names:", df.columns.tolist())

        balance_columns = st.multiselect(
            "Select columns to balance",
            options=df.columns.tolist(),
            default=df.columns.tolist()[:1]
        )

        if not balance_columns:
            st.error("Please select at least one column to balance")
            return

        num_territories = st.number_input(
            "Number of territories",
            min_value=2,
            max_value=len(df),
            value=2
        )

        if st.button("Create Territories"):
            with st.spinner("Distributing territories..."):
                territories = create_balanced_territories(df[balance_columns], num_territories)

            st.subheader("Territory Metrics")
            for i, territory in enumerate(territories):
                st.write(f"Territory {i+1} ({len(territory)} accounts)")
                st.write(territory.describe())

            combined = pd.concat(territories, ignore_index=True)
            st.markdown(get_download_link(combined, "all_territories.csv"), unsafe_allow_html=True)

def get_download_link(df: pd.DataFrame, filename: str) -> str:
    """Generate a CSV download link for a DataFrame."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}</a>'

if __name__ == "__main__":
    main()
