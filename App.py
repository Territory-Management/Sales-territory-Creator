import streamlit as st
import pandas as pd
import numpy as np
import base64
import logging
from typing import List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(file) -> Optional[pd.DataFrame]:
    """Load CSV file with error handling for encoding issues."""
    try:
        df = pd.read_csv(file, encoding='utf-8', engine='python', sep=None)
        logging.info("File successfully loaded with utf-8 encoding.")
        return df
    except UnicodeDecodeError:
        try:
            file.seek(0)
            df = pd.read_csv(file, encoding='latin1', engine='python', sep=None)
            logging.info("File successfully loaded with latin1 encoding.")
            return df
        except Exception as e:
            logging.error(f"Error loading file: {str(e)}")
            st.error("Error loading file. Please check the format.")
            return None
    except Exception as e:
        logging.error(f"Error loading file: {str(e)}")
        st.error("Error loading file. Please check the format.")
        return None

def normalize_numeric_column(series: pd.Series) -> pd.Series:
    """Normalize columns with monetary symbols and non-numeric characters."""
    return pd.to_numeric(
        series.astype(str).str.replace(r'[\u20ac$\u00a3,]', '', regex=True).str.replace(',', '.').str.strip(),
        errors='coerce'
    )

def distribute_equally_by_count_and_sum(df: pd.DataFrame, num_territories: int, balance_columns: List[str]) -> List[pd.DataFrame]:
    """Distribute rows equally by count and sum of specified columns across territories."""
    territories = [pd.DataFrame(columns=df.columns) for _ in range(num_territories)]
    sorted_df = df.sort_values(balance_columns, ascending=False)

    for _, row in sorted_df.iterrows():
        min_index = min(range(num_territories), key=lambda i: (len(territories[i]), -territories[i][balance_columns].sum().sum()))
        territories[min_index] = territories[min_index].append(row)

    return territories

def distribute_termination_clients_equally(territories: List[pd.DataFrame], termination_clients: pd.DataFrame) -> List[pd.DataFrame]:
    """Distribute termination clients equally among territories."""
    for i, client in enumerate(termination_clients.itertuples(index=False)):
        territories[i % len(territories)] = pd.concat([territories[i % len(territories)], pd.DataFrame([client._asdict()])], ignore_index=True)
    return territories

def create_territories(
    df: pd.DataFrame,
    num_territories: int,
    balance_columns: List[str],
    weights: Optional[List[float]] = None,
    years: Optional[List[int]] = None
) -> tuple[List[pd.DataFrame], pd.DataFrame]:
    """Create balanced territories with adjustments for disparities."""
    working_df = df.copy()
    termination_clients = pd.DataFrame()

    if years:
        working_df['Dt resiliation contrat all'] = pd.to_datetime(working_df['Dt resiliation contrat all'], errors='coerce')
        termination_clients = working_df[working_df['Dt resiliation contrat all'].dt.year.isin(years)]
        working_df = working_df[~working_df['Dt resiliation contrat all'].dt.year.isin(years)]

    for col in balance_columns:
        working_df[col] = normalize_numeric_column(working_df[col])

    if weights is None:
        weights = [1 / len(balance_columns)] * len(balance_columns)

    territories = distribute_equally_by_count_and_sum(working_df, num_territories, balance_columns)

    if not termination_clients.empty:
        territories = distribute_termination_clients_equally(territories, termination_clients)

    for i, territory in enumerate(territories):
        territory['Territory'] = i + 1

    metrics = []
    for i, territory in enumerate(territories):
        metric = {'Territory': i + 1, 'Count': len(territory)}
        for col in balance_columns:
            metric[f'{col}_total'] = territory[col].sum()
        metrics.append(metric)

    return territories, pd.DataFrame(metrics)

def get_download_link(df: pd.DataFrame, filename: str) -> str:
    """Generate a CSV download link."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download {filename}</a>'

def main():
    st.title('EquiTerritory')

    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
    if uploaded_file:
        df = load_data(uploaded_file)

        if df is not None:
            st.write("Data Preview:", df.head())
            st.write("Available Columns:", df.columns.tolist())

            balance_columns = st.multiselect(
                "Select columns to balance",
                options=df.columns.tolist(),
                default=df.columns.tolist()[:1]
            )

            if not balance_columns:
                st.error("Please select at least one column to balance")
                return

            col1, col2 = st.columns(2)
            with col1:
                num_territories = st.number_input(
                    "Number of territories", min_value=2, max_value=len(df), value=2
                )

            with col2:
                use_weights = st.checkbox("Use custom weights", value=False)

            years = st.multiselect(
                "Select termination years",
                options=[2024, 2025, 2026],
                default=[2024]
            )

            weights = None
            if use_weights:
                st.write("Assign absolute weights (weights will be normalized automatically):")
                weights = [st.number_input(f"Weight for {col}", min_value=0.0, value=100.0, step=10.0) for col in balance_columns]
                total = sum(weights)
                weights = [w / total for w in weights]

            if st.button("Create Territories"):
                territories, metrics = create_territories(
                    df, num_territories, balance_columns, weights, years
                )

                st.subheader("Territory Metrics")
                st.write(metrics)

                combined_territories = pd.concat(territories, ignore_index=True)

                st.markdown(
                    get_download_link(combined_territories, "territory_distribution.csv"),
                    unsafe_allow_html=True
                )

                for i, territory in enumerate(territories):
                    with st.expander(f"Territory {i+1} ({len(territory)} accounts)"):
                        st.write(territory)
                        st.markdown(
                            get_download_link(territory, f"territory_{i+1}.csv"),
                            unsafe_allow_html=True
                        )

if __name__ == "__main__":
    main()
