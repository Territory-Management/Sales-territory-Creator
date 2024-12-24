import streamlit as st
import pandas as pd
import numpy as np
import base64
import chardet
import plotly.express as px
import plotly.graph_objs as go
from typing import List, Optional, Dict, Any

# Machine Learning and Advanced Analytics
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.stats import entropy
import numba

class TerritoryBalancer:
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the Territory Balancer with comprehensive features
        """
        self.original_df = df.copy()
        self.df = df.copy()
        self.num_territories = 2
        self.balance_columns = []
        self.constraints = []
        
    def preprocess_data(self, balance_columns: List[str]):
        """
        Advanced data preprocessing
        """
        # Normalize and handle different data types
        self.balance_columns = balance_columns
        processed_df = self.df.copy()
        
        for col in balance_columns:
            # Handle different numeric formats
            processed_df[col] = pd.to_numeric(
                processed_df[col].astype(str)
                .str.replace(r'[€$£,]', '', regex=True)
                .str.replace(',', '.'),
                errors='coerce'
            )
        
        # Remove rows with NaN in balance columns
        processed_df.dropna(subset=balance_columns, inplace=True)
        
        # Standardize features
        scaler = StandardScaler()
        processed_columns = scaler.fit_transform(processed_df[balance_columns])
        
        return processed_df, processed_columns
    
    def advanced_distribution_strategy(
        self, 
        processed_df: pd.DataFrame, 
        processed_columns: np.ndarray
    ) -> List[pd.DataFrame]:
        """
        Multi-strategy territory distribution
        """
        # Hybrid approach combining multiple strategies
        territories = []
        
        # K-Means Clustering
        kmeans = KMeans(n_clusters=self.num_territories, random_state=42)
        cluster_labels = kmeans.fit_predict(processed_columns)
        processed_df['Cluster'] = cluster_labels
        
        # Entropy-based refinement
        for cluster in range(self.num_territories):
            cluster_data = processed_df[processed_df['Cluster'] == cluster]
            
            # Entropy sorting
            cluster_data['entropy_score'] = cluster_data[self.balance_columns].apply(
                lambda row: entropy(row), axis=1
            )
            
            # Sort and distribute
            sorted_cluster = cluster_data.sort_values('entropy_score', ascending=False)
            territories.append(sorted_cluster)
        
        return territories
    
    def validate_territories(self, territories: List[pd.DataFrame]) -> bool:
        """
        Validate territory distribution
        """
        metrics = []
        for i, territory in enumerate(territories):
            territory_metrics = {
                'Territory': i + 1,
                'Total Clients': len(territory)
            }
            
            for col in self.balance_columns:
                territory_metrics[f'{col}_Total'] = territory[col].sum()
                territory_metrics[f'{col}_Mean'] = territory[col].mean()
            
            metrics.append(territory_metrics)
        
        # Check distribution balance
        df_metrics = pd.DataFrame(metrics)
        variation_threshold = 0.2  # 20% variation allowed
        
        for col in self.balance_columns:
            total_col = f'{col}_Total'
            mean = df_metrics[total_col].mean()
            max_allowed = mean * (1 + variation_threshold)
            min_allowed = mean * (1 - variation_threshold)
            
            if any((df_metrics[total_col] > max_allowed) | 
                   (df_metrics[total_col] < min_allowed)):
                return False
        
        return True
    
    def visualize_territories(self, territories: List[pd.DataFrame]):
        """
        Create comprehensive territory visualization
        """
        metrics = []
        for i, territory in enumerate(territories):
            territory_metrics = {
                'Territory': i + 1,
                'Total Clients': len(territory)
            }
            
            for col in self.balance_columns:
                territory_metrics[f'{col}_Total'] = territory[col].sum()
                territory_metrics[f'{col}_Mean'] = territory[col].mean()
            
            metrics.append(territory_metrics)
        
        df_metrics = pd.DataFrame(metrics)
        
        # Plotly visualizations
        fig1 = px.bar(
            df_metrics, 
            x='Territory', 
            y='Total Clients', 
            title='Clients per Territory'
        )
        
        fig2 = go.Figure()
        for col in self.balance_columns:
            fig2.add_trace(go.Bar(
                x=df_metrics['Territory'],
                y=df_metrics[f'{col}_Total'],
                name=f'{col} Total'
            ))
        fig2.update_layout(title='Total Values per Territory')
        
        return fig1, fig2

def main():
    st.title('Advanced Territory Balancer 🌐')
    
    # File Upload
    uploaded_file = st.file_uploader(
        "Upload CSV File", 
        type=['csv'], 
        help="Upload your client data CSV file"
    )
    
    if uploaded_file:
        # Load and detect encoding
        try:
            raw_data = uploaded_file.read()
            uploaded_file.seek(0)
            encoding = chardet.detect(raw_data)['encoding']
            
            df = pd.read_csv(uploaded_file, encoding=encoding)
            st.success(f"File loaded successfully with {encoding} encoding")
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return
        
        # Display basic data info
        st.write("Data Overview")
        st.dataframe(df.head())
        
        # Columns selection
        balance_columns = st.multiselect(
            "Select Columns for Territory Balancing",
            options=df.select_dtypes(include=[np.number]).columns.tolist()
        )
        
        # Territory count
        num_territories = st.slider(
            "Number of Territories", 
            min_value=2, 
            max_value=10, 
            value=3
        )
        
        # Advanced Options
        with st.expander("Advanced Configuration"):
            st.write("Fine-tune Territory Distribution")
            
            # Variation Tolerance
            variation_tolerance = st.slider(
                "Variation Tolerance (%)", 
                min_value=5, 
                max_value=50, 
                value=20
            ) / 100
        
        # Distribution Button
        if st.button("Create Territories"):
            if not balance_columns:
                st.warning("Please select at least one numeric column")
                return
            
            # Initialize Balancer
            balancer = TerritoryBalancer(df)
            balancer.num_territories = num_territories
            
            # Preprocess Data
            processed_df, processed_columns = balancer.preprocess_data(balance_columns)
            
            # Distribute Territories
            territories = balancer.advanced_distribution_strategy(
                processed_df, 
                processed_columns
            )
            
            # Validate Territories
            is_valid = balancer.validate_territories(territories)
            
            if is_valid:
                st.success("Territories created successfully!")
                
                # Visualizations
                fig1, fig2 = balancer.visualize_territories(territories)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(fig1)
                with col2:
                    st.plotly_chart(fig2)
                
                # Download Options
                for i, territory in enumerate(territories, 1):
                    with st.expander(f"Territory {i} Details"):
                        st.dataframe(territory)
                        
                        # Download CSV
                        csv = territory.to_csv(index=False)
                        st.download_button(
                            label=f"Download Territory {i} CSV",
                            data=csv,
                            file_name=f'territory_{i}.csv',
                            mime='text/csv'
                        )
            else:
                st.warning("Could not create balanced territories. Try adjusting parameters.")

if __name__ == "__main__":
    main()
