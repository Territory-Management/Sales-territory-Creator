import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
from datetime import datetime
import plotly.express as px

def normalize_numeric_column(series: pd.Series) -> pd.Series:
    """Normalise les colonnes numériques en retirant les symboles monétaires"""
    return pd.to_numeric(
        series.astype(str)
        .str.replace(r'[\u20ac$\u00a3,]', '', regex=True)
        .str.replace(',', '.')
        .str.strip(),
        errors='coerce'
    )

def prepare_data(
    df: pd.DataFrame,
    balance_columns: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Prépare les données en séparant clients actifs et résiliés"""
    # Conversion de la date de résiliation
    df['Dt resiliation contrat all'] = pd.to_datetime(
        df['Dt resiliation contrat all'], 
        errors='coerce'
    )
    
    # Séparation des clients
    termination_clients = df[df['Dt resiliation contrat all'].notna()].copy()
    active_clients = df[df['Dt resiliation contrat all'].isna()].copy()
    
    # Normalisation des colonnes à équilibrer
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
    """Calcule les scores pour la répartition des territoires"""
    if weights is None:
        weights = [1.0 / len(balance_columns)] * len(balance_columns)
        
    df = df.copy()
    df['score'] = sum(
        df[col] * weight 
        for col, weight in zip(balance_columns, weights)
        if col != 'Dt resiliation contrat all'
    )
    return df.sort_values('score', ascending=False)

def distribute_territories(
    scored_df: pd.DataFrame,
    num_territories: int
) -> List[pd.DataFrame]:
    """Distribue les clients en territoires selon leurs scores"""
    return [
        scored_df.iloc[i::num_territories].copy() 
        for i in range(num_territories)
    ]

def distribute_terminations(
    territories: List[pd.DataFrame],
    termination_clients: pd.DataFrame
) -> List[pd.DataFrame]:
    """Distribue équitablement les clients avec résiliation"""
    # Trier les résiliations par date
    termination_clients = termination_clients.sort_values('Dt resiliation contrat all')
    
    # Distribution cyclique des résiliations
    for i, client in enumerate(termination_clients.itertuples(index=False)):
        territory_idx = i % len(territories)
        client_df = pd.DataFrame([client._asdict()])
        territories[territory_idx] = pd.concat(
            [territories[territory_idx], client_df],
            ignore_index=True
        )
    
    return territories

def calculate_metrics(
    territories: List[pd.DataFrame],
    balance_columns: List[str]
) -> pd.DataFrame:
    """Calcule les métriques pour chaque territoire"""
    metrics = []
    for i, territory in enumerate(territories):
        metric = {
            'Territory': i + 1,
            'Total_Clients': len(territory),
            'Resiliations': len(territory[territory['Dt resiliation contrat all'].notna()])
        }
        
        # Calculer les sommes pour chaque colonne d'équilibrage
        for col in balance_columns:
            if col != 'Dt resiliation contrat all':
                metric[f'{col}_Total'] = territory[col].sum()
                metric[f'{col}_Moyenne'] = territory[col].mean()
                metric[f'{col}_Mediane'] = territory[col].median()
                
        metrics.append(metric)
    
    return pd.DataFrame(metrics)

def create_balanced_territories(
    df: pd.DataFrame,
    num_territories: int,
    balance_columns: List[str],
    weights: Optional[List[float]] = None
) -> Tuple[List[pd.DataFrame], pd.DataFrame]:
    """Fonction principale de création des territoires équilibrés"""
    
    # Préparation des données
    active_clients, termination_clients = prepare_data(df, balance_columns)
    
    # Calcul des scores et distribution initiale
    scored_actives = calculate_territory_scores(
        active_clients,
        balance_columns,
        weights
    )
    territories = distribute_territories(scored_actives, num_territories)
    
    # Distribution des résiliations
    if not termination_clients.empty:
        territories = distribute_terminations(territories, termination_clients)
    
    # Ajout des numéros de territoire
    for i, territory in enumerate(territories):
        territory['Territory'] = i + 1
    
    # Calcul des métriques
    metrics = calculate_metrics(territories, balance_columns)
    
    return territories, metrics

def visualize_territories(territories: List[pd.DataFrame], metrics: pd.DataFrame):
    """Crée des visualisations pour analyser la répartition"""
    # Distribution des clients par territoire
    fig_clients = px.bar(
        metrics,
        x='Territory',
        y='Total_Clients',
        title='Distribution des clients par territoire'
    )
    
    # Distribution des résiliations
    fig_resil = px.bar(
        metrics,
        x='Territory',
        y='Resiliations',
        title='Distribution des résiliations par territoire'
    )
    
    return fig_clients, fig_resil

def export_territory_data(
    territories: List[pd.DataFrame],
    filename_prefix: str = "territoire"
) -> List[str]:
    """Exporte les données de chaque territoire"""
    filenames = []
    for i, territory in enumerate(territories, 1):
        filename = f"{filename_prefix}_{i}.csv"
        territory.to_csv(filename, index=False, encoding='utf-8-sig')
        filenames.append(filename)
    return filenames
