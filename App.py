import streamlit as st
import pandas as pd
import base64
import logging
import re
from typing import List, Optional
from datetime import datetime
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clean_numeric_value(value) -> float:
    """Convert a value to a float, cleaning any non-numeric characters."""
    try:
        return float(re.sub(r"[^\d.-]", "", str(value)))
    except (ValueError, TypeError):
        return 0.0

def load_data(file) -> Optional[pd.DataFrame]:
    """Load CSV data and preprocess it."""
    try:
        file.seek(0)
        sample_data = file.read(1024).decode("utf-8")

        # Infer the delimiter using csv.Sniffer
        import csv
        dialect = csv.Sniffer().sniff(sample_data)
        separator = dialect.delimiter

        # Read CSV with specified dtype for the first column to preserve leading zeros
        file.seek(0)
        df = pd.read_csv(file, sep=separator, dtype={0: str}, encoding="utf-8")

        # Ensure the first column is named correctly if unnamed
        if df.columns[0].startswith("Unnamed:"):
            df.rename(columns={df.columns[0]: "Client ID"}, inplace=True)

        return df
    except Exception as e:
        logging.error(f"Error loading file: {str(e)}")
        st.error(f"Error loading file: {str(e)}")
        return None

def distribute_territories(df: pd.DataFrame, num_territories: int, balance_columns: List[str]) -> List[pd.DataFrame]:
    """Distribute clients into territories, balancing both count and total value."""
    territories = [pd.DataFrame(columns=df.columns) for _ in range(num_territories)]

    # Calculate total values for balancing columns
    df["_total"] = df[balance_columns].applymap(clean_numeric_value).sum(axis=1)

    # Shuffle and sort by total value
    df_sorted = df.sample(frac=1, random_state=42).sort_values("_total", ascending=False).reset_index(drop=True)

    territory_sums = np.zeros(num_territories)
    territory_counts = np.zeros(num_territories)

    total_clients = len(df_sorted)

    # Initial round-robin distribution
    for idx, row in df_sorted.iterrows():
        territories[idx % num_territories] = pd.concat([territories[idx % num_territories], pd.DataFrame([row])], ignore_index=True)
        territory_counts[idx % num_territories] += 1
        territory_sums[idx % num_territories] += row["_total"]

    # Adjust distribution dynamically
    for idx, row in df_sorted.iterrows():
        weight_count = 0.9 if idx < len(df_sorted) * 0.3 else 0.5
        weight_sum = 1 - weight_count

        scores = [
            (
                territory_counts[i] * weight_count +
                territory_sums[i] * weight_sum +
                (float('inf') if territory_counts[i] > total_clients / num_territories * 1.1 else 0)
            )
            for i in range(num_territories)
        ]
        min_idx = np.argmin(scores)

        territories[min_idx] = pd.concat([territories[min_idx], pd.DataFrame([row])], ignore_index=True)
        territory_counts[min_idx] += 1
        territory_sums[min_idx] += row["_total"]

    # Final rebalancing to ensure equity
    for i in range(num_territories):
        territories[i] = territories[i].drop(columns=["_total"])
        territories[i].insert(0, "Territory", i + 1)

    logging.info(f"Total territories created: {len(territories)}")
    return territories

def get_territory_metrics(territories: List[pd.DataFrame], balance_columns: List[str]) -> pd.DataFrame:
    """Calculate metrics for each territory."""
    metrics = []
    for i, territory in enumerate(territories):
        metric = {"Territory": i + 1, "Count": len(territory)}
        for col in balance_columns:
            total = territory[col].apply(clean_numeric_value).sum()
            metric[f"{col}_total"] = total
        metrics.append(metric)
    return pd.DataFrame(metrics)

def get_download_link(df: pd.DataFrame, filename: str) -> str:
    """Generate a CSV download link for a DataFrame."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download {filename}</a>'

def main():
    st.title("Territory Distribution Tool")

    uploaded_file = st.file_uploader("Upload CSV file", type="csv")
    if uploaded_file:
        df = load_data(uploaded_file)

        if df is not None:
            st.write("Data Preview:", df.head())
            st.write("Column names:", df.columns.tolist())

            # Select balance columns
            available_columns = [col for col in df.columns if col != "Client ID"]
            st.write("Available columns for balancing:", available_columns)

            balance_columns = st.multiselect(
                "Select columns to balance",
                options=available_columns,
                default=available_columns[:1] if available_columns else []
            )

            if not balance_columns:
                st.error("Please select at least one column to balance")
                return

            # Number of territories
            num_territories = st.number_input(
                "Number of territories",
                min_value=2,
                max_value=len(df),
                value=2
            )

            if st.button("Create Territories"):
                with st.spinner("Distributing territories..."):
                    territories = distribute_territories(df, num_territories, balance_columns)

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
