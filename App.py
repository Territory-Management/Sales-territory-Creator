import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_data(file) -> pd.DataFrame:
    """Load the CSV file."""
    try:
        raw_data = file.read()
        file.seek(0)
        return pd.read_csv(file)
    except Exception as e:
        st.error(f"Error loading the file: {str(e)}")
        return pd.DataFrame()

def normalize_data(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Normalize specified columns in the DataFrame."""
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
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

        st.subheader("📊 Territory Metrics")
        st.dataframe(metrics, use_container_width=True)

        # Download links
        st.subheader("📥 Download")
        combined = pd.concat(territories, ignore_index=True)
        combined['Territory'] = combined['Territory'].astype(int)
        csv = combined.to_csv(index=False, encoding='utf-8-sig')
        b64 = base64.b64encode(csv.encode()).decode()
        st.markdown(f'<a href="data:file/csv;base64,{b64}" download="all_territories.csv">Download All Territories</a>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
