import streamlit as st
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from sklearn.cluster import KMeans
from pulp import *
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TerritoryMetrics:
    territory_id: int
    clients: List[int]  # Client indices
    metrics: Dict[str, float]  # Column sums

class TerritoryOptimizer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.num_territories = 2
        self.balance_columns: List[str] = []
        
    def preprocess_data(self, balance_columns: List[str]) -> pd.DataFrame:
        """Clean and prepare data for optimization."""
        processed_df = self.df.copy()
        
        for col in balance_columns:
            processed_df[col] = pd.to_numeric(
                processed_df[col].astype(str).str.replace(r'[^\d.-]', '', regex=True),
                errors='coerce'
            )
            
        processed_df.dropna(subset=balance_columns, inplace=True)
        return processed_df
        
    def optimize_territories(self, balance_columns: List[str]) -> List[TerritoryMetrics]:
        """Create balanced territories using linear programming."""
        processed_df = self.preprocess_data(balance_columns)
        n_clients = len(processed_df)
        target_size = n_clients // self.num_territories
        
        # Initialize optimization problem
        prob = LpProblem("Territory_Balancing", LpMinimize)
        
        # Decision variables: x[i,j] = 1 if client i is in territory j
        x = LpVariable.dicts("client_assignment",
                           ((i, j) for i in range(n_clients) 
                            for j in range(self.num_territories)),
                           cat='Binary')
        
        # Objective: Minimize variance between territories
        territory_sums = {}
        for col in balance_columns:
            values = processed_df[col].values
            for j in range(self.num_territories):
                territory_sums[(col,j)] = lpSum(values[i] * x[i,j] 
                                              for i in range(n_clients))
        
        # Target values
        targets = {col: processed_df[col].sum() / self.num_territories 
                  for col in balance_columns}
        
        # Add objective function
        prob += lpSum((territory_sums[(col,j)] - targets[col])**2 
                     for col in balance_columns 
                     for j in range(self.num_territories))
        
        # Constraints
        # Each client must be assigned to exactly one territory
        for i in range(n_clients):
            prob += lpSum(x[i,j] for j in range(self.num_territories)) == 1
            
        # Each territory should have approximately equal size
        for j in range(self.num_territories):
            prob += lpSum(x[i,j] for i in range(n_clients)) >= target_size - 1
            prob += lpSum(x[i,j] for i in range(n_clients)) <= target_size + 1
        
        # Solve the optimization problem
        prob.solve(PULP_CBC_CMD(msg=False))
        
        # Extract results
        territories = []
        for j in range(self.num_territories):
            clients = [i for i in range(n_clients) 
                      if value(x[i,j]) > 0.5]
            
            metrics = {col: processed_df.iloc[clients][col].sum() 
                      for col in balance_columns}
            
            territories.append(TerritoryMetrics(
                territory_id=j+1,
                clients=clients,
                metrics=metrics
            ))
        
        return territories

def main():
    st.set_page_config(page_title="Territory Optimizer", layout="wide")
    st.title('Territory Optimizer')
    
    uploaded_file = st.file_uploader("Upload CSV File", type=['csv'])
    
    if uploaded_file:
        try:
            # Handle CSV with inconsistent columns
            df = pd.read_csv(uploaded_file, on_bad_lines='warn', engine='python')
            
            # Log information about malformed rows
            if df.shape[0] == 0:
                st.error("No valid data rows found in CSV")
                return
                
            st.info(f"Successfully loaded {df.shape[0]} rows and {df.shape[1]} columns")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            balance_columns = st.multiselect(
                "Select Columns for Balancing",
                options=numeric_cols
            )
            
            num_territories = st.slider("Number of Territories", 2, 10, 2)
            
            if st.button("Optimize Territories"):
                optimizer = TerritoryOptimizer(df)
                optimizer.num_territories = num_territories
                
                territories = optimizer.optimize_territories(balance_columns)
                
                for territory in territories:
                    with st.expander(f"Territory {territory.territory_id}"):
                        territory_df = df.iloc[territory.clients]
                        st.write(f"Clients: {len(territory.clients)}")
                        for col, value in territory.metrics.items():
                            st.write(f"{col}: {value:,.2f}")
                        st.dataframe(territory_df)
                        
                        csv = territory_df.to_csv(index=False)
                        st.download_button(
                            f"Download Territory {territory.territory_id}",
                            data=csv,
                            file_name=f'territory_{territory.territory_id}.csv',
                            mime='text/csv'
                        )
                        
        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
