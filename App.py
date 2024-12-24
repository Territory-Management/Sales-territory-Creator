import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import calendar

def main():
    st.set_page_config(layout="wide")
    
    with st.sidebar:
        st.image("https://your-logo.png", width=200)
        st.title("Territory Balancer")
        uploaded_file = st.file_uploader("📂 Fichier CSV", type="csv")

    if not uploaded_file:
        st.markdown("""
        <div style='text-align: center; padding: 50px;'>
            <h1>👋 Bienvenue sur Territory Balancer</h1>
            <p>Importez votre fichier CSV pour commencer</p>
        </div>
        """, unsafe_allow_html=True)
        return

    df = load_data(uploaded_file)
    if df is None:
        return

    # Configuration des colonnes
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("📊 Aperçu des données")
        st.dataframe(df.head(), use_container_width=True)
        
    with col2:
        st.subheader("🎯 Configuration")
        num_territories = st.number_input("Nombre de territoires", 2, 20, 4)
        balance_columns = st.multiselect(
            "Colonnes à équilibrer",
            options=[c for c in df.columns if c != 'Dt resiliation contrat all'],
            default=df.select_dtypes(include=['float64', 'int64']).columns[:1].tolist()
        )

    # Gestion des résiliations
    st.subheader("📅 Gestion des résiliations")
    col3, col4 = st.columns(2)
    
    with col3:
        df['Dt resiliation contrat all'] = pd.to_datetime(df['Dt resiliation contrat all'], errors='coerce')
        resiliation_stats = analyze_terminations(df)
        
        st.metric("Clients avec résiliation", 
                 f"{resiliation_stats['total_terminations']} ({resiliation_stats['termination_percentage']:.1f}%)")
        
        # Graphique des résiliations par mois
        fig = px.bar(resiliation_stats['monthly_data'], 
                    title="Résiliations par mois",
                    labels={'value': 'Nombre', 'month': 'Mois'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col4:
        resiliation_strategy = st.radio(
            "Stratégie de distribution des résiliations",
            ["Équitable", "Par date", "Par valeur"],
            help="Comment répartir les clients avec résiliation"
        )
        
        if resiliation_strategy == "Par date":
            max_per_territory = st.slider(
                "Max. résiliations par territoire/mois",
                1, 20, 5
            )

    if st.button("🚀 Générer les territoires", type="primary"):
        territories, metrics = create_territories(
            df=df,
            num_territories=num_territories,
            balance_columns=balance_columns,
            resiliation_strategy=resiliation_strategy,
            max_terminations_per_territory=(max_per_territory if resiliation_strategy == "Par date" else None)
        )
        
        display_results(territories, metrics)

def analyze_terminations(df: pd.DataFrame) -> dict:
    """Analyse les statistiques de résiliation"""
    termination_data = df[df['Dt resiliation contrat all'].notna()]
    total = len(df)
    total_terminations = len(termination_data)
    
    monthly_data = (
        termination_data['Dt resiliation contrat all']
        .dt.to_period('M')
        .value_counts()
        .sort_index()
        .reset_index()
    )
    monthly_data['month'] = monthly_data['index'].astype(str)
    
    return {
        'total_terminations': total_terminations,
        'termination_percentage': (total_terminations / total * 100),
        'monthly_data': monthly_data
    }

def display_results(territories: list, metrics: pd.DataFrame):
    """Affiche les résultats avec graphiques"""
    st.subheader("📈 Résultats")
    
    # Métriques principales
    cols = st.columns(len(territories))
    for i, (col, territory) in enumerate(zip(cols, territories)):
        with col:
            st.metric(
                f"Territoire {i+1}",
                f"{len(territory)} clients",
                f"{len(territory[territory['Dt resiliation contrat all'].notna()])} résiliations"
            )
    
    # Graphiques de distribution
    for col in metrics.columns:
        if col not in ['Territory', 'Count']:
            fig = px.bar(metrics, 
                        x='Territory', 
                        y=col,
                        title=f"Distribution de {col}")
            st.plotly_chart(fig, use_container_width=True)
    
    # Export des données
    st.download_button(
        "⬇️ Télécharger tous les territoires",
        pd.concat(territories).to_csv(index=False).encode('utf-8'),
        "territoires.csv",
        "text/csv"
    )
