import streamlit as st
import pandas as pd
import numpy as np
import base64
import chardet
from typing import List, Optional, Tuple

def load_data(file) -> Optional[pd.DataFrame]:
    """Load the CSV file while detecting the encoding."""
    try:
        raw_data = file.read()
        file.seek(0)
        encoding = chardet.detect(raw_data)['encoding']
        df = pd.read_csv(file, encoding=encoding, engine='python', sep=None)
        return df
    except Exception as e:
        st.error(f"Error loading the file: {str(e)}")
        return None

def normalize_numeric_column(series: pd.Series) -> pd.Series:
    """Normalize numeric columns."""
    return pd.to_numeric(
        series.astype(str)
        .str.replace(r'[\u20ac$\u00a3,]', '', regex=True)
        .str.replace(',', '.')
        .str.strip(),
        errors='coerce'
    )

def prepare_data(df: pd.DataFrame, balance_columns: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare data by separating active and terminated clients."""
    df['Dt resiliation contrat all'] = pd.to_datetime(df['Dt resiliation contrat all'], errors='coerce')
    termination_clients = df[df['Dt resiliation contrat all'].notna()].copy()
    active_clients = df[df['Dt resiliation contrat all'].isna()].copy()
    
    for col in balance_columns:
        if col != 'Dt resiliation contrat all':
            active_clients[col] = normalize_numeric_column(active_clients[col])
            termination_clients[col] = normalize_numeric_column(termination_clients[col])
            
    return active_clients, termination_clients

def calculate_scores(df: pd.DataFrame, balance_columns: List[str], weights: Optional[List[float]]) -> pd.Series:
    """Calculate scores using vectorized operations."""
    weighted_sums = df[balance_columns].fillna(0).values @ np.array(weights)
    return pd.Series(weighted_sums, index=df.index)

def distribute_territories(df: pd.DataFrame, num_territories: int) -> List[pd.DataFrame]:
    """Distribute clients into territories ensuring equity."""
    territories = [[] for _ in range(num_territories)]
    territory_sums = np.zeros(num_territories)

    for _, client in df.iterrows():
        min_sum_idx = np.argmin(territory_sums)
        territories[min_sum_idx].append(client)
        territory_sums[min_sum_idx] += client['score']

    return [pd.DataFrame(territory) for territory in territories]

def calculate_metrics(territories: List[pd.DataFrame], balance_columns: List[str]) -> pd.DataFrame:
    """Calculate territory metrics."""
    metrics = []
    for i, territory in enumerate(territories):
        metric = {
            'Territory': i + 1,
            'Total_Clients': len(territory),
            'Resiliations': len(territory[territory['Dt resiliation contrat all'].notna()])
        }
        
        for col in balance_columns:
            if col != 'Dt resiliation contrat all':
                metric[f'{col}_Total'] = territory[col].sum()
                metric[f'{col}_Average'] = territory[col].mean()
        metrics.append(metric)
    
    return pd.DataFrame(metrics)

def create_balanced_territories(
    df: pd.DataFrame,
    num_territories: int,
    balance_columns: List[str],
    weights: Optional[List[float]] = None
) -> Tuple[List[pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    """Create balanced territories with optimized distribution."""
    active_clients, termination_clients = prepare_data(df, balance_columns)

    # Calculate scores for both active and terminated clients
    if weights is None:
        weights = [1.0 / len(balance_columns)] * len(balance_columns)
    
    active_clients['score'] = calculate_scores(active_clients, balance_columns, weights)
    termination_clients['score'] = calculate_scores(termination_clients, balance_columns, weights)

    # Distribute all clients
    all_clients = pd.concat([active_clients, termination_clients]).sort_values('score', ascending=False)
    territories = distribute_territories(all_clients, num_territories)
    
    for i, territory in enumerate(territories):
        territory['Territory'] = i + 1

    metrics = calculate_metrics(territories, balance_columns)
    combined_territories = pd.concat(territories, ignore_index=True)
    combined_territories['Territory'] = combined_territories['Territory'].astype(int)

    return territories, metrics, combined_territories

def get_download_link(df: pd.DataFrame, filename: str) -> str:
    """Generate a download link."""
    csv = df.to_csv(index=False, encoding='utf-8-sig')
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download {filename}</a>'

def display_territory_metrics(metrics: pd.DataFrame):
    """Display territory metrics with Streamlit."""
    st.subheader("📊 Territory Metrics")
    cols = st.columns(len(metrics))
    
    for i, (_, row) in enumerate(metrics.iterrows()):
        with cols[i]:
            st.metric(
                f"Territory {row['Territory']}", 
                f"{row['Total_Clients']} clients",
                f"{row['Resiliations']} resiliations"
            )
    
    st.dataframe(metrics, use_container_width=True)

def main():
    st.set_page_config(layout="wide")
    st.title('Territory Balancer')

    uploaded_file = st.file_uploader("📂 Upload a CSV file", type="csv")
    if not uploaded_file:
        st.info("Upload your CSV file to get started")
        return

    df = load_data(uploaded_file)
    if df is None:
        return

    col1, col2 = st.columns([2, 1])
    with col1:
        st.dataframe(df.head(), use_container_width=True)
    
    with col2:
        num_territories = st.number_input(
            "Number of Territories", 
            min_value=2, 
            max_value=len(df), 
            value=4
        )
        
        balance_columns = st.multiselect(
            "Columns to Balance",
            options=[c for c in df.columns if c != 'Dt resiliation contrat all'],
            default=df.select_dtypes(include=['float64', 'int64']).columns[:1].tolist()
        )

        use_weights = st.checkbox("Use Weights")
        weights = None
        if use_weights:
            weights = [
                st.number_input(f"Weight for {col}", value=1.0, min_value=0.0)
                for col in balance_columns
            ]
            total = sum(weights)
            weights = [w / total for w in weights]

    if st.button("🚀 Generate Territories", type="primary"):
        territories, metrics, combined_territories = create_balanced_territories(
            df, num_territories, balance_columns, weights
        )
        
        display_territory_metrics(metrics)
        
        st.subheader("📥 Download")
        # Global download
        st.markdown(
            get_download_link(combined_territories, "all_territories.csv"),
            unsafe_allow_html=True
        )
        
        # Individual downloads
        st.subheader("Download by Territory")
        for i, territory in enumerate(territories, 1):
            st.markdown(
                get_download_link(territory, f"territory_{i}.csv"),
                unsafe_allow_html=True
            )

if __name__ == "__main__":
    main()
