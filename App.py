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
        series.astype(str).str.replace(r'[€$£,]', '', regex=True).str.replace(',', '.').str.strip(),
        errors='coerce'
    )


def filter_by_termination_date(df: pd.DataFrame, years: List[int]) -> pd.DataFrame:
    """Filtrer les lignes où la date de résiliation correspond aux années spécifiées"""
    # Vérifier si la colonne 'Dt resiliation contrat all' existe
    if 'Dt resiliation contrat all' not in df.columns:
        st.error("La colonne 'Dt resiliation contrat all' est introuvable dans les données.")
        return df  # Retourner le DataFrame sans modification

    # Convertir la colonne 'Dt resiliation contrat all' en datetime
    df['Dt resiliation contrat all'] = pd.to_datetime(df['Dt resiliation contrat all'], errors='coerce')

    # Filtrer par les années spécifiées
    return df[df['Dt resiliation contrat all'].dt.year.isin(years)]


def calculate_weights(balance_columns: List[str]) -> List[float]:
    """Calculer les poids normalisés pour les colonnes choisies"""
    weights = []
    for col in balance_columns:
        weight = st.slider(f"Poids pour {col}", 0.0, 1.0, 1.0 / len(balance_columns), 0.1)
        weights.append(weight)
    # Normaliser les poids pour qu'ils somment à 1
    total = sum(weights)
    return [w / total for w in weights]


def balance_territories(territories: List[pd.DataFrame], balance_columns: List[str]) -> List[pd.DataFrame]:
    """Redistribue les éléments entre les territoires pour minimiser les écarts"""
    # Calculer les sommes des colonnes pour chaque territoire
    sums = {i: territory[balance_columns].sum() for i, territory in enumerate(territories)}

    # Calculer l'écart total
    diff = {i: sum(abs(sums[i] - sums[j])) for i in sums for j in sums if i != j}

    # Redistribution simple : échange les éléments entre les territoires pour équilibrer les sommes
    for i in range(len(territories)):
        for j in range(i + 1, len(territories)):
            if diff.get(i, 0) > diff.get(j, 0):
                # Échanger des éléments entre les territoires
                transfer_element = territories[i].iloc[0]  # Éléments les plus "lourds"
                territories[i] = territories[i].drop(territories[i].index[0])
                territories[j] = territories[j].append(transfer_element)
    
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

    # Si des années de résiliation sont spécifiées, filtrer les données
    if years:
        working_df = filter_by_termination_date(working_df, years)

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

    # Rééquilibrage pour minimiser les écarts
    territories = balance_territories(territories, balance_columns)

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
                weights = calculate_weights(balance_columns)

            # Créer les territoires lorsque l'utilisateur clique sur le bouton
            if st.button("Créer les territoires"):
                territories, metrics = create_territories(
                    df, num_territories, balance_columns, weights, years
                )

                # Afficher les métriques des territoires
                st.subheader("Métriques des territoires")
                st.write(metrics)

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
