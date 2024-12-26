import streamlit as st
import pandas as pd
import base64
import logging
import re
from typing import List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clean_numeric_value(value) -> float:
    """
    Clean numeric values by removing non-numeric characters and converting to float.
    """
    if pd.isna(value):
        return 0.0
    try:
        str_value = str(value)
        clean_value = re.sub(r'[^\d.-]', '', str_value)
        return float(clean_value) if clean_value else 0.0
    except (ValueError, TypeError):
        return 0.0

def load_data(file) -> Optional[pd.DataFrame]:
    """
    Load and preprocess the CSV file. Ensures the first column is correctly interpreted.
    """
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

def distribute_territories(df: pd.DataFrame, num_territories: int, balance_columns: List[str]) -> List[pd.DataFrame]:
    """
    Distribute rows to territories, balancing both total sums and row counts.
    """
    # Initialize empty territories
    territories = [pd.DataFrame(columns=df.columns) for _ in range(num_territories)]
    territory_sums = [0.0] * num_territories
    territory_counts = [0] * num_territories

    # Calculate total values for the balance columns
    df['_total'] = df.apply(lambda row: sum(clean_numeric_value(row[col]) for col in balance_columns), axis=1)

    # Sort the dataframe by total value in descending order
    df_sorted = df.sort_values('_total', ascending=False)

    # Hybrid assignment considering both sums and counts
    for _, row in df_sorted.iterrows():
        # Find the territory with the smallest adjusted load
        min_idx = min(
            range(num_territories),
            key=lambda i: (
                territory_sums[i] + 0.5 * (territory_counts[i] / (len(df) / num_territories))
            )
        )
        territories[min_idx] = pd.concat([territories[min_idx], pd.DataFrame([row])], ignore_index=True)
        territory_sums[min_idx] += row['_total']
        territory_counts[min_idx] += 1

    # Remove the temporary '_total' column from the final territories
    for i in range(num_territories):
        territories[i] = territories[i].drop(columns='_total')
        territories[i].insert(0, 'Territory', i + 1)

    # Log final distribution
    logging.info("Final Distribution of Territories:")
    for idx, (total, count) in enumerate(zip(territory_sums, territory_counts)):
        logging.info(f"Territory {idx + 1}: Total Sum = {total}, Row Count = {count}")

    return territories

def get_territory_metrics(territories: List[pd.DataFrame], balance_columns: List[str]) -> pd.DataFrame:
    """
    Generate metrics for each territory, including count and total values for specified columns.
    """
    metrics = []
    for i, territory in enumerate(territories):
        metric = {
            'Territory': i + 1,
            'Count': len(territory)
        }
        for col in balance_columns:
            total = territory[col].apply(clean_numeric_value).sum()
            metric[f'{col}_total'] = total
        metrics.append(metric)
    return pd.DataFrame(metrics)

def get_download_link(df: pd.DataFrame, filename: str) -> str:
    """
    Generate a download link for a dataframe as a CSV file.
    """
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
                options=[col for col in df.columns if col != 'Code client' and col != 'Dt resiliation contrat all'],
                default=[col for col in df.columns if col != 'Code client' and col != 'Dt resiliation contrat all'][:1]
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
