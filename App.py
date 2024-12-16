import streamlit as st
import pandas as pd
import numpy as np
import base64
import chardet
from typing import List, Optional

def load_data(file) -> Optional[pd.DataFrame]:
    try:
        # Détecter l'encodage du fichier
        raw_data = file.read()
        file.seek(0)
        encoding = chardet.detect(raw_data)['encoding']
        
        # Lire le fichier avec l'encodage détecté et un moteur Python pour mieux détecter les délimiteurs
        df = pd.read_csv(file, encoding=encoding, engine='python', sep=None)
        return df
    except Exception as e:
        # Essayer avec des encodages courants
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'windows-1252']
        for enc in encodings:
            try:
                file.seek(0)
                df = pd.read_csv(file, encoding=enc, engine='python', sep=None)
                return df
            except:
                continue
        st.error(f"Erreur lors du chargement du fichier : {str(e)}")
        return None

def normalize_numeric_column(series: pd.Series) -> pd.Series:
    """ Normaliser une colonne contenant des valeurs numériques avec des symboles monétaires et des séparateurs """
    try:
        # Supprimer les symboles monétaires et les virgules, puis convertir en numérique
        return pd.to_numeric(
            series.astype(str).str.replace(r'[€$£,]', '', regex=True).str.replace(',', '.'),
            errors='coerce'
        )
    except:
        return series

def create_territories(
    df: pd.DataFrame,
    num_territories: int,
    balance_columns: List[str],
    weights: Optional[List[float]] = None
) -> tuple[List[pd.DataFrame], pd.DataFrame]:
    # Créer une copie de travail de la DataFrame
    working_df = df.copy()
    
    # Normaliser les colonnes
    for col in balance_columns:
        working_df[col] = normalize_numeric_column(working_df[col])
    
    # Calculer le score pondéré
    if weights is None:
        weights = [1/len(balance_columns)] * len(balance_columns)
    
    # Calculer la somme des scores
    working_df['score'] = sum(
        working_df[col] * weight 
        for col, weight in zip(balance_columns, weights)
    )
    
    # Trier les données par score et répartir dans les territoires
    working_df = working_df.sort_values('score', ascending=False)
    territories = [working_df.iloc[i::num_territories] for i in range(num_territories)]
    
    # Calculer les métriques pour chaque territoire
    metrics = []
    for i, territory in enumerate(territories):
        metric = {'Territory': i + 1, 'Count': len(territory)}
        for col in balance_columns:
            metric[f'{col}_total'] = territory[col].sum()
        metrics.append(metric)
    
    return territories, pd.DataFrame(metrics)

def get_download_link(df: pd.DataFrame, filename: str) -> str:
    """ Générer un lien de téléchargement pour un DataFrame au format CSV """
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
            st.write("Colonnes disponibles :", df.columns.tolist())  # Liste de toutes les colonnes

            # Sélectionner les colonnes à équilibrer (toutes les colonnes)
            balance_columns = st.multiselect(
                "Sélectionner les colonnes à équilibrer",
                options=df.columns.tolist(),
                default=df.columns.tolist()[:1]
            )

            # Interface pour entrer le nombre de territoires et choisir les poids
            col1, col2 = st.columns(2)
            with col1:
                num_territories = st.number_input(
                    "Nombre de territoires",
                    min_value=2,
                    max_value=len(df),
                    value=2
                )
            
            with col2:
                use_weights = st.checkbox("Utiliser des poids personnalisés", value=False)

            # Gérer les poids personnalisés
            weights = None
            if use_weights and balance_columns:
                weights = []
                for col in balance_columns:
                    weight = st.slider(
                        f"Poids pour {col}",
                        0.0, 1.0, 1.0/len(balance_columns),
                        0.1
                    )
                    weights.append(weight)
                
                # Normaliser les poids pour qu'ils somment à 1
                total = sum(weights)
                weights = [w / total for w in weights]

            # Créer les territoires quand l'utilisateur clique sur le bouton
            if st.button("Créer les territoires"):
                if not balance_columns:
                    st.error("Veuillez sélectionner au moins une colonne à équilibrer")
                    return

                # Appeler la fonction pour créer les territoires
                territories, metrics = create_territories(
                    df, 
                    num_territories, 
                    balance_columns,
                    weights
                )

                # Afficher les métriques des territoires
                st.subheader("Métriques des territoires")
                st.write(metrics)

                # Afficher les territoires avec des liens de téléchargement
                for i, territory in enumerate(territories):
                    with st.expander(f"Territoire {i+1} ({len(territory)} comptes)") :
                        st.write(territory)
                        st.markdown(
                            get_download_link(territory, f"territoire_{i+1}.csv"),
                            unsafe_allow_html=True
                        )

if __name__ == "__main__":
    main()
