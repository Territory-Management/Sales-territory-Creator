import streamlit as st
import pandas as pd
import numpy as np
import csv
from io import StringIO
from typing import List

def load_data(file) -> pd.DataFrame:
    """Load the CSV file and handle rows with varying numbers of fields."""
    try:
        raw_data = file.read().decode('utf-8')
        file.seek(0)
        
        # Use csv.reader to read the file
        reader = csv.reader(StringIO(raw_data))
        rows = list(reader)
        
        # Determine the maximum number of columns
        max_columns = max(len(row) for row in rows)
        
        # Standardize row lengths by expanding each row to the maximum number of columns
        standardized_rows = [row + [None] * (max_columns - len(row)) for row in rows]
        
        # Create a DataFrame using the standardized rows
        df = pd.DataFrame(standardized_rows[1:], columns=standardized_rows[0])
        
        # Convert all numerical columns to appropriate data types
        df = df.apply(pd.to_numeric, errors='ignore')
        
        return df
    except Exception as e:
        st.error(f"Error loading the file: {str(e)}")
        return pd.DataFrame()

def normalize_data(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Normalize specified columns in the DataFrame using min-max scaling."""
    for col in columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            min_col = df[col].min()
            max_col = df[col].max()
            df[col] = (df[col] - min_col) / (max_col - min_col)
        else:
            st.warning(f"Column '{col}' is not numeric and will be ignored in normalization.")
    return df

def calculate_scores(df: pd.DataFrame, balance_columns: List[str]) -> pd.Series:
    """Calculate scores using equal weights for each column."""
    weights = np.ones(len(balance_columns)) / len(balance_columns)
    scores = df[balance_columns].fillna(0).dot(weights)
    return scores

def distribute_territories(df: pd.DataFrame, num_territories: int) -> List[pd.DataFrame]:
    """Distribute clients into territories ensuring equity."""
    territories = [pd.DataFrame(columns=df.columns) for _ in range(num_territories)]
    territory_totals = np.zeros(num_territories)

    # Sort by score to distribute high-value clients first
    df = df.sort_values('score', ascending=False)

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
            'Resiliations': territory['Dt resiliation contrat all'].notna().sum() if 'Dt resiliation contrat all' in territory else 0
        }
        
        for col in balance_columns:
            if col in territory:
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
        options=[c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])],
        default=df.select_dtypes(include=[np.number]).columns.tolist()
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
