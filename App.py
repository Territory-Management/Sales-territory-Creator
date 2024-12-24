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

def create_balanced_territories(
    df: pd.DataFrame,
    num_territories: int,
    balance_columns: List[str],
    years: Optional[List[int]] = None
) -> tuple[List[pd.DataFrame], pd.DataFrame]:
    """
    Crée des territoires équilibrés en utilisant une approche gloutonne améliorée
    
    Args:
    - df: DataFrame source
    - num_territories: Nombre de territoires à créer
    - balance_columns: Colonnes à utiliser pour l'équilibrage
    - years: Années de résiliation à filtrer (optionnel)
    
    Returns:
    - Liste des territoires
    - DataFrame des métriques des territoires
    """
    # Créer une copie de travail du DataFrame
    working_df = df.copy()

    # Filtrer par années de résiliation si spécifiées
    if years:
        working_df['Dt resiliation contrat all'] = pd.to_datetime(working_df['Dt resiliation contrat all'], errors='coerce')
        working_df = working_df[~working_df['Dt resiliation contrat all'].dt.year.isin(years)]

    # Normaliser les colonnes numériques
    for col in balance_columns:
        working_df[col] = normalize_numeric_column(working_df[col])

    # Calculer un score global pour chaque ligne
    working_df['global_score'] = working_df[balance_columns].sum(axis=1)

    # Trier par score global décroissant
    working_df = working_df.sort_values('global_score', ascending=False)

    # Initialiser les territoires
    territories = [[] for _ in range(num_territories)]
    territory_totals = [0] * num_territories

    # Algorithme de répartition gloutonne
    for _, row in working_df.iterrows():
        # Trouver le territoire avec le total le plus bas
        min_territory_index = territory_totals.index(min(territory_totals))
        
        # Ajouter la ligne au territoire avec le total le plus bas
        territories[min_territory_index].append(row)
        
        # Mettre à jour le total de ce territoire
        territory_totals[min_territory_index] += row['global_score']

    # Convertir les territoires en DataFrames
    territory_dfs = [pd.DataFrame(territory) for territory in territories]

    # Ajouter une colonne "Territory"
    for i, territory in enumerate(territory_dfs):
        territory['Territory'] = i + 1

    # Calculer les métriques
    metrics = []
    for i, territory in enumerate(territory_dfs):
        metric = {'Territory': i + 1, 'Count': len(territory)}
        for col in balance_columns:
            metric[f'{col}_total'] = territory[col].sum()
            metric[f'{col}_mean'] = territory[col].mean()
        metrics.append(metric)

    return territory_dfs, pd.DataFrame(metrics)

def get_download_link(df: pd.DataFrame, filename: str) -> str:
    """Génère un lien de téléchargement CSV"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">Télécharger {filename}</a>'

def main():
    st.title('Équilibreur de Territoires Avancé')

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

            # Interface pour entrer le nombre de territoires
            num_territories = st.number_input(
                "Nombre de territoires", min_value=2, max_value=len(df), value=2
            )

            # Sélectionner plusieurs années de résiliation pour filtrer
            years = st.multiselect(
                "Sélectionner les années de résiliation à exclure",
                options=[2024, 2025, 2026],
                default=[]
            )

            # Créer les territoires lorsque l'utilisateur clique sur le bouton
            if st.button("Créer les territoires"):
                territories, metrics = create_balanced_territories(
                    df, num_territories, balance_columns, years
                )

                # Afficher les métriques des territoires
                st.subheader("Métriques des territoires")
                st.dataframe(metrics)

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
