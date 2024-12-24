import streamlit as st
import pandas as pd
import numpy as np
import base64
import logging
from typing import List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(file) -> Optional[pd.DataFrame]:
    """Charge le fichier CSV avec gestion des erreurs d'encodage."""
    try:
        df = pd.read_csv(file, encoding='utf-8', engine='python', sep=None)
        logging.info("Fichier chargé avec succès avec l'encodage utf-8.")
        return df
    except UnicodeDecodeError:
        try:
            file.seek(0)
            df = pd.read_csv(file, encoding='latin1', engine='python', sep=None)
            logging.info("Fichier chargé avec succès avec l'encodage latin1.")
            return df
        except Exception as e:
            logging.error(f"Erreur lors du chargement du fichier : {str(e)}")
            st.error("Erreur lors du chargement du fichier. Veuillez vérifier le format.")
            return None
    except Exception as e:
        logging.error(f"Erreur lors du chargement du fichier : {str(e)}")
        st.error("Erreur lors du chargement du fichier. Veuillez vérifier le format.")
        return None

def normalize_numeric_column(series: pd.Series) -> pd.Series:
    """Normalise les colonnes avec des symboles monétaires et des caractères non numériques."""
    return pd.to_numeric(
        series.astype(str).str.replace(r'[\u20ac$\u00a3,]', '', regex=True).str.replace(',', '.').str.strip(),
        errors='coerce'
    )

def distribute_termination_clients(df: pd.DataFrame, termination_clients: pd.DataFrame, num_territories: int) -> List[pd.DataFrame]:
    """Ventile les clients avec une date de résiliation équitablement parmi les territoires."""
    territories = [df.iloc[i::num_territories] for i in range(num_territories)]
    logging.info(f"Distribution de {len(termination_clients)} clients résiliés.")
    
    for i, client in enumerate(termination_clients.itertuples(index=False)):
        territories[i % num_territories] = pd.concat([territories[i % num_territories], pd.DataFrame([client._asdict()])], ignore_index=True)
    
    return territories

def create_territories(
    df: pd.DataFrame,
    num_territories: int,
    balance_columns: List[str],
    weights: Optional[List[float]] = None,
    years: Optional[List[int]] = None
) -> tuple[List[pd.DataFrame], pd.DataFrame]:
    """Crée des territoires équilibrés avec un ajustement des écarts."""
    working_df = df.copy()
    termination_clients = pd.DataFrame()

    if years:
        working_df['Dt resiliation contrat all'] = pd.to_datetime(working_df['Dt resiliation contrat all'], errors='coerce')
        termination_clients = working_df[working_df['Dt resiliation contrat all'].dt.year.isin(years)]
        working_df = working_df[~working_df['Dt resiliation contrat all'].dt.year.isin(years)]

    for col in balance_columns:
        working_df[col] = normalize_numeric_column(working_df[col])

    if weights is None:
        weights = [1 / len(balance_columns)] * len(balance_columns)

    working_df['score'] = sum(working_df[col] * weight for col, weight in zip(balance_columns, weights))
    working_df = working_df.sort_values('score', ascending=False)
    territories = [working_df.iloc[i::num_territories] for i in range(num_territories)]

    if not termination_clients.empty:
        territories = distribute_termination_clients(pd.concat(territories, ignore_index=True), termination_clients, num_territories)

    for i, territory in enumerate(territories):
        territory['Territory'] = i + 1

    metrics = []
    for i, territory in enumerate(territories):
        metric = {'Territory': i + 1, 'Count': len(territory)}
        for col in balance_columns:
            metric[f'{col}_total'] = territory[col].sum()
        metrics.append(metric)

    return territories, pd.DataFrame(metrics)

def get_download_link(df: pd.DataFrame, filename: str) -> str:
    """Génère un lien de téléchargement CSV."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">Télécharger {filename}</a>'

def main():
    st.title('EquiTerritory')

    uploaded_file = st.file_uploader("Téléchargez un fichier CSV", type="csv")
    if uploaded_file:
        df = load_data(uploaded_file)

        if df is not None:
            st.write("Aperçu des données :", df.head())
            st.write("Colonnes disponibles :", df.columns.tolist())

            balance_columns = st.multiselect(
                "Sélectionner les colonnes à équilibrer",
                options=df.columns.tolist(),
                default=df.columns.tolist()[:1]
            )

            if not balance_columns:
                st.error("Veuillez sélectionner au moins une colonne à équilibrer")
                return

            col1, col2 = st.columns(2)
            with col1:
                num_territories = st.number_input(
                    "Nombre de territoires", min_value=2, max_value=len(df), value=2
                )

            with col2:
                use_weights = st.checkbox("Utiliser des poids personnalisés", value=False)

            years = st.multiselect(
                "Sélectionner les années de résiliation",
                options=[2024, 2025, 2026],
                default=[2024]
            )

            weights = None
            if use_weights:
                st.write("Attribuer des poids absolus (les poids seront normalisés automatiquement) :")
                weights = [st.number_input(f"Poids pour {col}", min_value=0.0, value=100.0, step=10.0) for col in balance_columns]
                total = sum(weights)
                weights = [w / total for w in weights]

            if st.button("Créer les territoires"):
                territories, metrics = create_territories(
                    df, num_territories, balance_columns, weights, years
                )

                st.subheader("Métriques des territoires")
                st.write(metrics)

                combined_territories = pd.concat(territories, ignore_index=True)

                st.markdown(
                    get_download_link(combined_territories, "repartition_territoires.csv"),
                    unsafe_allow_html=True
                )

                for i, territory in enumerate(territories):
                    with st.expander(f"Territoire {i+1} ({len(territory)} comptes)"):
                        st.write(territory)
                        st.markdown(
                            get_download_link(territory, f"territoire_{i+1}.csv"),
                            unsafe_allow_html=True
                        )

if __name__ == "__main__":
    main()
