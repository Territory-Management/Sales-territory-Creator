import streamlit as st
import pandas as pd
import numpy as np
import base64
import chardet
from typing import List, Optional

def load_data(file) -> Optional[pd.DataFrame]:
    """Charge le fichier CSV en détectant l'encodage et gère les erreurs"""
    try:
        # Détection de l'encodage
        raw_data = file.read()
        file.seek(0)
        encoding = chardet.detect(raw_data)['encoding']
        
        # Lecture du CSV avec l'encodage détecté
        df = pd.read_csv(file, encoding=encoding, engine='python', sep=None)
        return df
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier : {str(e)}")
        return None

def normalize_numeric_column(series: pd.Series) -> pd.Series:
    """Normalise les colonnes avec des symboles monétaires et des caractères non numériques"""
    return pd.to_numeric(
        series.astype(str).str.replace(r'[\u20ac$\u00a3,]', '', regex=True).str.replace(',', '.').str.strip(),
        errors='coerce'
    )

def distribute_termination_clients(df: pd.DataFrame, termination_clients: pd.DataFrame, num_territories: int) -> List[pd.DataFrame]:
    """Ventile les clients avec une date de résiliation équitablement parmi les territoires"""
    territories = [df.iloc[i::num_territories] for i in range(num_territories)]
    
    # Ajout des clients avec résiliation équitablement
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
    """Crée des territoires équilibrés avec un ajustement des écarts"""
    # Créer une copie de travail du DataFrame
    working_df = df.copy()

    # Si des années de résiliation sont spécifiées, séparer les clients
    termination_clients = pd.DataFrame()
    if years:
        working_df['Dt resiliation contrat all'] = pd.to_datetime(working_df['Dt resiliation contrat all'], errors='coerce')
        termination_clients = working_df[working_df['Dt resiliation contrat all'].dt.year.isin(years)]
        working_df = working_df[~working_df['Dt resiliation contrat all'].dt.year.isin(years)]

    # Normaliser les colonnes
    for col in balance_columns:
        working_df[col] = normalize_numeric_column(working_df[col])

    # Si les poids ne sont pas définis, les répartir uniformément
    if weights is None:
        weights = [1 / len(balance_columns)] * len(balance_columns)

    # Calculer le score pondéré
    working_df['score'] = sum(working_df[col] * weight for col, weight in zip(balance_columns, weights))

    # Trier les données par score décroissant
    working_df = working_df.sort_values('score', ascending=False)

    # Répartir les données en territoires
    territories = [working_df.iloc[i::num_territories] for i in range(num_territories)]

    # Ventiler les clients avec résiliation équitablement
    if not termination_clients.empty:
        territories = distribute_termination_clients(pd.concat(territories, ignore_index=True), termination_clients, num_territories)

    # Ajouter une colonne "Territory" pour identifier chaque territoire
    for i, territory in enumerate(territories):
        territory['Territory'] = i + 1

    # Calcul des métriques pour chaque territoire
    metrics = []
    for i, territory in enumerate(territories):
        metric = {'Territory': i + 1, 'Count': len(territory)}
        for col in balance_columns:
            metric[f'{col}_total'] = territory[col].sum()
        metrics.append(metric)

    return territories, pd.DataFrame(metrics)

def get_download_link(df: pd.DataFrame, filename: str) -> str:
    """Génère un lien de téléchargement CSV"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">Télécharger {filename}</a>'

def main():
    st.title('Fast Territory Balancer')

    uploaded_file = st.file_uploader("Téléchargez un fichier CSV", type="csv")
    if uploaded_file:
        df = load_data(uploaded_file)

        if df is not None:
            st.write("Aperçu des données :", df.head())
            st.write("Colonnes disponibles :", df.columns.tolist())

            # Sélectionner les colonnes à équilibrer
            balance_columns = st.multiselect(
                "Sélectionner les colonnes à équilibrer",
                options=df.columns.tolist(),
                default=df.columns.tolist()[:1]
            )

            if not balance_columns:
                st.error("Veuillez sélectionner au moins une colonne à équilibrer")
                return

            # Interface pour entrer le nombre de territoires et choisir les poids
            col1, col2 = st.columns(2)
            with col1:
                num_territories = st.number_input(
                    "Nombre de territoires", min_value=2, max_value=len(df), value=2
                )

            with col2:
                use_weights = st.checkbox("Utiliser des poids personnalisés", value=False)

            # Sélectionner plusieurs années de résiliation pour filtrer
            years = st.multiselect(
                "Sélectionner les années de résiliation",
                options=[2024, 2025, 2026],
                default=[2024]
            )

            # Calculer les poids si nécessaire
            weights = None
            if use_weights:
                st.write("Attribuer des poids absolus (les poids seront normalisés automatiquement) :")
                weights = [st.number_input(f"Poids pour {col}", min_value=0.0, value=100.0, step=10.0) for col in balance_columns]
                total = sum(weights)
                weights = [w / total for w in weights]

            # Créer les territoires lorsque l'utilisateur clique sur le bouton
            if st.button("Créer les territoires"):
                territories, metrics = create_territories(
                    df, num_territories, balance_columns, weights, years
                )

                # Afficher les métriques des territoires
                st.subheader("Métriques des territoires")
                st.write(metrics)

                # Fusionner tous les territoires en un seul DataFrame avec leur numéro de territoire
                combined_territories = pd.concat(territories, ignore_index=True)

                # Lien pour télécharger le fichier combiné
                st.markdown(
                    get_download_link(combined_territories, "repartition_territoires.csv"),
                    unsafe_allow_html=True
                )

                # Afficher les territoires avec des liens de téléchargement
                for i, territory in enumerate(territories):
                    with st.expander(f"Territoire {i+1} ({len(territory)} comptes)"):
                        st.write(territory)
                        st.markdown(
                            get_download_link(territory, f"territoire_{i+1}.csv"),
                            unsafe_allow_html=True
                        )

if __name__ == "__main__":
    main()
