import streamlit as st
import pandas as pd
import numpy as np
import base64
import csv
from io import StringIO
from typing import List

def load_data(file) -> pd.DataFrame:
    """Load and clean the CSV file to handle parsing errors."""
    try:
        raw_data = file.read().decode('utf-8')
        file.seek(0)
        
        # Use csv.reader to manually filter out malformed lines
        reader = csv.reader(StringIO(raw_data), delimiter=',')
        cleaned_data = []
        expected_columns = None
        
        for i, row in enumerate(reader):
            if i == 0:
                expected_columns = len(row)  # Determine the number of columns expected
                cleaned_data.append(row)     # Include the header
            elif len(row) == expected_columns:
                cleaned_data.append(row)
            else:
                st.warning(f"Skipping malformed line {i+1}: {row}")

        # Convert cleaned data back to a CSV string
        cleaned_csv = StringIO()
        writer = csv.writer(cleaned_csv, delimiter=',')
        writer.writerows(cleaned_data)
        cleaned_csv.seek(0)
        
        # Load the DataFrame from the cleaned CSV string
        df = pd.read_csv(cleaned_csv)
        return df
    except Exception as e:
        st.error(f"Error loading the file: {str(e)}")
        return pd.DataFrame()

def normalize_data(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Normalize specified columns in the DataFrame using min-max scaling."""
    for col in columns:
        min_col = df[col].min()
        max_col = df[col].max()
        df[col] = (df[col] - min_col) / (max_col - min_col)
    return df

def calculate_scores(df: pd.DataFrame, balance_columns: List[str]) -> pd.Series:
    """Calculate scores using equal weights for each column."""
    weights = np.ones(len(balance_columns)) / len(balance_columns)
    scores = df[balance_columns].dot(weights)
    return scores

def distribute_territories(df: pd.DataFrame, num_territories: int) -> List[pd.DataFrame]:
    """Distribute clients into territories ensuring equity."""
    territories = [pd.DataFrame(columns=df.columns) for _ in range(num_territories)]
    territory_totals = np.zeros(num_territories)

    for _, client in df.iterrows():
        min_idx = np.argmin(territory_totals)
        territories[min_idx] = pd.concat([territories[min_idx], pd.DataFrame([client])], ignore_index=True)
        territory_totals[min_idx] += client['score']

    return territories

def calculate_metrics(territories: List[pd.DataFrame], balance_columns: List[str]) -> pd.DataFrame:
    """Calculate metrics for each territory."""
    metrics = []
    for i, territory in enumerate(territories):
        metric = {
            'Territory': i + 1,
            'Total Clients': len(territory),
            'Resiliations': territory['Dt resiliation contrat all'].notna().sum()
        }
        
        for col in balance_columns:
            metric[f'{col} Total'] = territory[col].sum()
            metric[f'{col} Average'] = territory[col].mean()
        metrics.append(metric)
    
    return pd.DataFrame(metrics)

def get_download_link(df: pd.DataFrame, filename: str) -> str:
    """Generate a download link for a DataFrame."""
    csv = df.to_csv(index=False, encoding='utf-8-sig')
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download {filename}</a>'

def display_territory_metrics(metrics: pd.DataFrame):
    """Display territory metrics with Streamlit."""
    st.subheader("📊 Territory Metrics")
    st.dataframe(metrics, use_container_width=True)

def main():
    st.set_page_config(layout="wide")
    st.title('Territory Balancer')

    uploaded_file = st.file_uploader("📂 Upload a CSV file", type="csv")
    if not uploaded_file:
        st.info("Upload your CSV file to get started")
        return

    df = load_data(uploaded_file)
    if df.empty:
        return

    balance_columns = st.multiselect(
        "Columns to Balance",
        options=[c for c in df.columns if c != 'Dt resiliation contrat all'],
        default=df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    )

    num_territories = st.number_input(
        "Number of Territories", 
        min_value=2, 
        max_value=min(len(df), 100), 
        value=4
    )

    # Normalize and calculate scores
    df = normalize_data(df, balance_columns)
    df['score'] = calculate_scores(df, balance_columns)

    if st.button("🚀 Generate Territories"):
        territories = distribute_territories(df, num_territories)
        metrics = calculate_metrics(territories, balance_columns)

        display_territory_metrics(metrics)

        st.subheader("📥 Download")
        combined = pd.concat(territories, ignore_index=True)
        combined['Territory'] = combined['Territory'].astype(int)
        st.markdown(get_download_link(combined, "all_territories.csv"), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
