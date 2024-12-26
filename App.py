import streamlit as st
import pandas as pd
import base64
import logging
import re
from typing import List, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Utility function to clean numeric values
def clean_numeric_value(value) -> float:
    if pd.isna(value):
        return 0.0
    try:
        str_value = str(value)
        clean_value = re.sub(r"[^\d.-]", "", str_value)
        return float(clean_value) if clean_value else 0.0
    except (ValueError, TypeError):
        return 0.0

# Function to load and preprocess data
def load_data(file) -> Optional[pd.DataFrame]:
    try:
        file.seek(0)
        sample_data = file.read(1024).decode("utf-8")
        import csv
        dialect = csv.Sniffer().sniff(sample_data)
        separator = dialect.delimiter

        file.seek(0)
        df = pd.read_csv(file, sep=separator, dtype=str, encoding="utf-8")

        if df.columns[0].startswith("Unnamed:"):
            df.rename(columns={df.columns[0]: "Code client"}, inplace=True)

        # Convert 'Dt resiliation' to datetime
        if "Dt resiliation contrat all" in df.columns:
            df["Dt resiliation contrat all"] = pd.to_datetime(df["Dt resiliation contrat all"], errors="coerce")

        return df
    except Exception as e:
        logging.error(f"Error loading file: {str(e)}")
        st.error(f"Error loading file: {str(e)}")
        return None

# Filter rows based on resiliation date
def filter_rows(df: pd.DataFrame) -> pd.DataFrame:
    current_year = datetime.now().year
    next_year = current_year + 1
    if "Dt resiliation contrat all" in df.columns:
        return df[(df["Dt resiliation contrat all"].dt.year > next_year) | df["Dt resiliation contrat all"].isna()]
    return df

# Distribute territories
def distribute_territories(df: pd.DataFrame, num_territories: int, balance_columns: List[str]) -> List[pd.DataFrame]:
    territories = [pd.DataFrame(columns=df.columns) for _ in range(num_territories)]

    # Calculate total values for balancing columns
    df["_total"] = df.apply(lambda row: sum(clean_numeric_value(row[col]) for col in balance_columns), axis=1)

    # Sort data by the total value for prioritization
    df_sorted = df.sort_values("_total", ascending=False)

    territory_sums = [0.0] * num_territories
    territory_counts = [0] * num_territories

    # Assign rows to territories in a balanced way
    for _, row in df_sorted.iterrows():
        min_idx = min(range(num_territories), key=lambda i: (
            territory_counts[i],  # Prioritize even row distribution
            territory_sums[i]  # Then consider total value
        ))
        territories[min_idx] = pd.concat([territories[min_idx], pd.DataFrame([row])], ignore_index=True)
        territory_counts[min_idx] += 1
        territory_sums[min_idx] += row["_total"]

    # Remove the helper '_total' column
    for i in range(num_territories):
        territories[i] = territories[i].drop(columns=["_total"])
        territories[i].insert(0, "Territory", i + 1)

    logging.info(f"Total territories created: {len(territories)}")
    return territories

# Metrics for balancing validation
def get_territory_metrics(territories: List[pd.DataFrame], balance_columns: List[str]) -> pd.DataFrame:
    metrics = []
    for i, territory in enumerate(territories):
        metric = {"Territory": i + 1, "Count": len(territory)}
        for col in balance_columns:
            total = territory[col].apply(clean_numeric_value).sum()
            metric[f"{col}_total"] = total
        metrics.append(metric)
    return pd.DataFrame(metrics)

# Function to create download link
def get_download_link(df: pd.DataFrame, filename: str) -> str:
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download {filename}</a>'

# Main Streamlit app
def main():
    st.title("Territory Distribution Tool")

    uploaded_file = st.file_uploader("Upload CSV file", type="csv")
    if uploaded_file:
        df = load_data(uploaded_file)

        if df is not None:
            st.write("Data Preview:", df.head())
            st.write("Column names:", df.columns.tolist())

            # Filter rows based on resiliation date
            df_filtered = filter_rows(df)
            st.write("Filtered Data Preview (Rows with valid resiliation dates):", df_filtered.head())

            # Select balance columns
            balance_columns = st.multiselect(
                "Select columns to balance",
                options=[col for col in df_filtered.columns if col not in ["Code client", "Dt resiliation contrat all"]],
                default=[col for col in df_filtered.columns if col not in ["Code client", "Dt resiliation contrat all"][:1]]
            )

            if not balance_columns:
                st.error("Please select at least one column to balance")
                return

            # Number of territories
            num_territories = st.number_input(
                "Number of territories",
                min_value=2,
                max_value=len(df_filtered),
                value=2
            )

            if st.button("Create Territories"):
                territories = distribute_territories(df_filtered, num_territories, balance_columns)
                metrics = get_territory_metrics(territories, balance_columns)

                # Display metrics
                st.subheader("Territory Metrics")
                st.write(metrics)

                # Combined territories
                combined = pd.concat(territories, ignore_index=True)
                st.markdown(get_download_link(combined, "all_territories.csv"), unsafe_allow_html=True)

                # Individual territory downloads
                for i, territory in enumerate(territories):
                    with st.expander(f"Territory {i+1} ({len(territory)} accounts)"):
                        st.write(territory)
                        st.markdown(
                            get_download_link(territory, f"territory_{i+1}.csv"),
                            unsafe_allow_html=True
                        )

if __name__ == "__main__":
    main()
