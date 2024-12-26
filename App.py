import streamlit as st
import pandas as pd
import base64
import logging
import re
from typing import List, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clean_numeric_value(value) -> float:
    """Cleans and converts a value to numeric, removing non-numeric characters."""
    if pd.isna(value):
        return 0.0
    try:
        str_value = str(value)
        clean_value = re.sub(r'[^\d.-]', '', str_value)
        return float(clean_value) if clean_value else 0.0
    except (ValueError, TypeError):
        return 0.0

def preprocess_dataframe(df: pd.DataFrame, balance_columns: List[str]) -> pd.DataFrame:
    """Prepares the DataFrame by cleaning numeric columns and handling missing data."""
    for col in balance_columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in the uploaded file.")
        df[col] = df[col].apply(clean_numeric_value)
    return df

def load_data(file) -> Optional[pd.DataFrame]:
    """Loads a CSV file and handles headers and separators."""
    try:
        file.seek(0)
        sample_data = file.read(1024).decode('utf-8')
        import csv
        dialect = csv.Sniffer().sniff(sample_data)
        separator = dialect.delimiter

        file.seek(0)
        df = pd.read_csv(file, sep=separator, dtype=str)

        if df.columns[0].startswith('Unnamed:'):
            df.rename(columns={df.columns[0]: 'Code client'}, inplace=True)

        return df

    except Exception as e:
        logging.error(f"Error loading file: {str(e)}")
        st.error(f"Error loading file: {str(e)}")
        return None

def distribute_territories(
    df: pd.DataFrame,
    num_territories: int,
    balance_columns: List[str]
) -> List[pd.DataFrame]:
    """Distributes rows across territories balancing count and sum of specified columns."""
    # Initialize empty territories
    territories = [list() for _ in range(num_territories)]
    territory_sums = [0.0] * num_territories
    territory_counts = [0] * num_territories

    # Sort rows by the total value of balance columns in descending order
    sorted_rows = df.apply(
        lambda row: sum(row[col] for col in balance_columns),
        axis=1
    ).sort_values(ascending=False).index

    # Assign rows using a bin-packing heuristic
    for idx in sorted_rows:
        row = df.loc[idx]
        row_total = sum(row[col] for col in balance_columns)

        # Determine the best territory to assign based on total sum and count
        best_territory = min(
            range(num_territories),
            key=lambda t: (territory_sums[t] + row_total, territory_counts[t])
        )
        territories[best_territory].append(row)
        territory_sums[best_territory] += row_total
        territory_counts[best_territory] += 1

    # Convert lists back to DataFrames
    territory_dfs = [pd.DataFrame(territory, columns=df.columns) for territory in territories]

    for i, territory in enumerate(territory_dfs):
        territory.insert(0, 'Territory', i + 1)

    return territory_dfs

def get_territory_metrics(territories: List[pd.DataFrame], balance_columns: List[str]) -> pd.DataFrame:
    """Calculates metrics for each territory."""
    metrics = []
    for i, territory in enumerate(territories):
        metric = {
            'Territory': i + 1,
            'Count': len(territory)
        }
        for col in balance_columns:
            metric[f'{col}_total'] = territory[col].sum()
        metrics.append(metric)
    return pd.DataFrame(metrics)

def get_download_link(df: pd.DataFrame, filename: str) -> str:
    """Generates a download link for a DataFrame."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download {filename}</a>'

def main():
    st.title('Territory Distribution Tool')

    uploaded_file = st.file_uploader("Upload CSV file", type="csv")
    if uploaded_file:
        df = load_data(uploaded_file)

        if df is not None:
            st.write("Data Preview:", df.head())
            st.write("Column names:", df.columns.tolist())

            balance_columns = st.multiselect(
                "Select columns to balance",
                options=[col for col in df.columns if col not in ('Code client', 'Dt resiliation contrat all')],
                default=[col for col in df.columns if col not in ('Code client', 'Dt resiliation contrat all')][:1]
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
                # Preprocess the DataFrame
                try:
                    df = preprocess_dataframe(df, balance_columns)
                except ValueError as e:
                    st.error(str(e))
                    return

                # Distribute territories
                territories = distribute_territories(df, num_territories, balance_columns)
                metrics = get_territory_metrics(territories, balance_columns)

                st.subheader("Territory Metrics")
                st.write(metrics)

                combined = pd.concat(territories, ignore_index=True)
                st.markdown(get_download_link(combined, "all_territories.csv"), unsafe_allow_html=True)

                for i, territory in enumerate(territories):
                    with st.expander(f"Territory {i+1} ({len(territory)} accounts)"):
                        st.write(territory)
                        st.markdown(
                            get_download_link(territory, f"territory_{i+1}.csv"),
                            unsafe_allow_html=True
                        )

if __name__ == "__main__":
    main()
