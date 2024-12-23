import streamlit as st
import pandas as pd
import numpy as np
import base64
import chardet
from typing import List, Optional, Tuple
from datetime import datetime

def load_data(file) -> Optional[pd.DataFrame]:
    """Charge le fichier CSV en détectant l'encodage"""
    try:
        raw_data = file.read()
        file.seek(0)
        encoding = chardet.detect(raw_data)['encoding']
        df = pd.read_csv(file, encoding=encoding, engine='python', sep=None)
        return df
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier : {str(e)}")
        return None

def normalize_numeric_column(series: pd.Series) -> pd.Series:
    """Normalise les colonnes numériques"""
    return pd.to_numeric(
        series.astype(str)
        .str.replace(r'[\u20ac$\u00a3,]', '', regex=True)
        .str.replace(',', '.')
        .str.strip(),
        errors='coerce'
    )

def prepare_data(df: pd.DataFrame, balance_columns: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Prépare les données en séparant clients actifs et résiliés"""
    df['Dt resiliation contrat all'] = pd.to_datetime(df['Dt resiliation contrat all'], errors='coerce')
    termination_clients = df[df['Dt resiliation contrat all'].notna()].copy()
    active_clients = df[df['Dt resiliation contrat all'].isna()].copy()
    
    for col in balance_columns:
        if col != 'Dt resiliation contrat all':
            active_clients[col] = normalize_numeric_column(active_clients[col])
            termination_clients[col] = normalize_numeric_column(termination_clients[col])
            
    return active_clients, termination_clients

def calculate_territory_scores(
    df: pd.DataFrame,
    balance_columns: List[str],
    weights: Optional[List[float]] = None
) -> pd.DataFrame:
    """Calcule les scores pour la répartition"""
    if weights is None:
        weights = [1.0 / len(balance_columns)] * len(balance_columns)
        
    df = df.copy()
    score_columns = [col for col in balance_columns if col != 'Dt resiliation contrat all']
    df['score'] = sum(df[col] * weight for col, weight in zip(score_columns, weights))
    return df.sort_values('score', ascending=False)

def distribute_territories(df: pd.DataFrame, num_territories: int) -> List[pd.DataFrame]:
    """Distribue les clients en territoires"""
    return [df.iloc[i::num_territories].copy() for i in range(num_territories)]

def distribute_terminations(territories: List[pd.DataFrame], termination_clients: pd.DataFrame) -> List[pd.DataFrame]:
    """Distribue les clients avec résiliation"""
    termination_clients = termination_clients.sort_values('Dt resiliation contrat all')
    
    for i, client in enumerate(termination_clients.itertuples(index=False)):
        territory_idx = i % len(territories)
        territories[territory_idx] = pd.concat(
            [territories[territory_idx], pd.DataFrame([client._asdict()])],
            ignore_index=True
        )
    
    return territories

def calculate_metrics(territories: List[pd.DataFrame], balance_columns: List[str]) -> pd.DataFrame:
    """Calcule les métriques des territoires"""
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
                metric[f'{col}_Moyenne'] = territory[col].mean()
        metrics.append(metric)
    
    return pd.DataFrame(metrics)

def create_balanced_territories(
    df: pd.DataFrame,
    num_territories: int,
    balance_columns: List[str],
    weights: Optional[List[float]] = None
) -> Tuple[List[pd.DataFrame], pd.DataFrame]:
    """Crée les territoires équilibrés"""
    active_clients, termination_clients = prepare_data(df, balance_columns)
    scored_actives = calculate_territory_scores(active_clients, balance_columns, weights)
    territories = distribute_territories(scored_actives, num_territories)
    
    if not termination_clients.empty:
        territories = distribute_terminations(territories, termination_clients)
    
    for i, territory in enumerate(territories):
        territory['Territory'] = i + 1
    
    metrics = calculate_metrics(territories, balance_columns)
    return territories, metrics

def get_download_link(df: pd.DataFrame, filename: str) -> str:
    """Génère un lien de téléchargement"""
    csv = df.to_csv(index=False, encoding='utf-8-sig')
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">Télécharger {filename}</a>'

def display_territory_metrics(metrics: pd.DataFrame):
    """Affiche les métriques avec Streamlit"""
    st.subheader("📊 Métriques par territoire")
    cols = st.columns(len(metrics))
    
    for i, (_, row) in enumerate(metrics.iterrows()):
        with cols[i]:
            st.metric(
                f"Territoire {row['Territory']}", 
                f"{row['Total_Clients']} clients",
                f"{row['Resiliations']} résiliations"
            )
    
    st.dataframe(metrics, use_container_width=True)

def main():
    st.set_page_config(layout="wide")
    st.title('Territory Balancer')

    uploaded_file = st.file_uploader("📂 Importer un fichier CSV", type="csv")
    if not uploaded_file:
        st.info("Importez votre fichier CSV pour commencer")
        return

    df = load_data(uploaded_file)
    if df is None:
        return

    col1, col2 = st.columns([2, 1])
    with col1:
        st.dataframe(df.head(), use_container_width=True)
    
    with col2:
        num_territories = st.number_input(
            "Nombre de territoires", 
            min_value=2, 
            max_value=len(df), 
            value=4
        )
        
        balance_columns = st.multiselect(
            "Colonnes à équilibrer",
            options=[c for c in df.columns if c != 'Dt resiliation contrat all'],
            default=df.select_dtypes(include=['float64', 'int64']).columns[:1].tolist()
        )

        use_weights = st.checkbox("Utiliser des poids")
        weights = None
        if use_weights:
            weights = [
                st.number_input(f"Poids pour {col}", value=1.0, min_value=0.0)
                for col in balance_columns
            ]
            total = sum(weights)
            weights = [w / total for w in weights]

    if st.button("🚀 Générer les territoires", type="primary"):
        territories, metrics = create_balanced_territories(
            df, num_territories, balance_columns, weights
        )
        
        display_territory_metrics(metrics)
        
        st.subheader("📥 Téléchargement")
        for i, territory in enumerate(territories, 1):
            st.markdown(
                get_download_link(territory, f"territoire_{i}.csv"),
                unsafe_allow_html=True
            )

if __name__ == "__main__":
    main()
