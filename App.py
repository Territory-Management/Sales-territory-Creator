import streamlit as st
import pandas as pd
import base64
import logging
import re
from typing import List, Optional
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clean_numeric_value(value: str) -> float:
    """Convert a value to a float, cleaning any non-numeric characters."""
    try:
        return float(re.sub(r"[^\d.-]", "", str(value)))
    except (ValueError, TypeError):
        return 0.0

def infer_separator(file) -> str:
    """Infer the CSV delimiter from the file content."""
    import csv
    sample_data = file.read(1024).decode("utf-8")
    file.seek(0)
    return csv.Sniffer().sniff(sample_data).delimiter

def load_data(file) -> Optional[pd.DataFrame]:
    """Load CSV data and preprocess it."""
    try:
        separator = infer_separator(file)
        df = pd.read_csv(file, sep=separator, dtype={0: str}, encoding="utf-8")
        if df.columns[0].startswith("Unnamed:"):
            df.rename(columns={df.columns[0]: "Client ID"}, inplace=True)
        return df
    except Exception as e:
        logging.error(f"Error loading file: {e}")
        st.error(f"Error loading file: {e}")
        return None

def refine_distribution(territories: List[pd.DataFrame], num_territories: int) -> List[pd.DataFrame]:
    """Refine the distribution of territories to better balance the total sums."""
    current_sums = np.array([territory["_total"].sum() for territory in territories])
    target_sum = current_sums.mean()

    for _ in range(10):  # Limit to a fixed number of iterations
        for i in range(num_territories):
            for j in range(num_territories):
                if i != j and current_sums[i] > target_sum and current_sums[j] < target_sum:
                    # Attempt to swap entries to balance the sums
                    for entry_idx in territories[i].index:
                        entry = territories[i].loc[entry_idx]
                        potential_new_sum_i = current_sums[i] - entry["_total"]
                        potential_new_sum_j = current_sums[j] + entry["_total"]

                        if abs(potential_new_sum_i - target_sum) + abs(potential_new_sum_j - target_sum) < abs(current_sums[i] - target_sum) + abs(current_sums[j] - target_sum):
                            # Perform the swap
                            territories[j] = pd.concat([territories[j], entry.to_frame().T], ignore_index=True)
                            territories[i] = territories[i].drop(entry_idx).reset_index(drop=True)

                            # Update sums
                            current_sums[i] = potential_new_sum_i
                            current_sums[j] = potential_new_sum_j

    return territories

def distribute_territories(df: pd.DataFrame, num_territories: int, balance_columns: List[str]) -> List[pd.DataFrame]:
    """Distribute clients into territories, balancing both count and total value."""
    # Calculate total values for balancing columns
    df["_total"] = df[balance_columns].applymap(clean_numeric_value).sum(axis=1)

    # Sort by total value
    df_sorted = df.sort_values("_total", ascending=False).reset_index(drop=True)

    # Initialize territories
    territories = [pd.DataFrame(columns=df.columns) for _ in range(num_territories)]
    territory_sums = np.zeros(num_territories)

    for idx, row in df_sorted.iterrows():
        # Assign to territory with the lowest total value
        min_idx = np.argmin(territory_sums)
        territories[min_idx] = pd.concat([territories[min_idx], pd.DataFrame([row])], ignore_index=True)
        territory_sums[min_idx] += row["_total"]

    # Perform refinement to improve balance
    territories = refine_distribution(territories, num_territories)

    # Ensure "Territory" column is added correctly
    for i in range(num_territories):
        territories[i] = territories[i].drop(columns=["_total"])
        territories[i] = territories[i].assign(Territory=(i + 1))

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
