import streamlit as st
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, value
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TerritoryMetrics:
    territory_id: int
    clients: List[int]
    metrics: Dict[str, float]
    active_clients: int
    inactive_clients: int

class TerritoryOptimizer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.num_territories = 2
        self.balance_columns: List[str] = []

    def preprocess_data(self, balance_columns: List[str], date_resiliation: str = None) -> pd.DataFrame:
        """Clean data and identify active/inactive clients."""
        processed_df = self.df.copy()

        # Handle active/inactive clients
        if date_resiliation and date_resiliation in processed_df.columns:
            processed_df['is_active'] = pd.to_datetime(processed_df[date_resiliation], errors='coerce').isna()
        else:
            processed_df['is_active'] = True

        # Ensure numeric columns
        valid_columns = []
        for col in balance_columns:
            try:
                processed_df[col] = pd.to_numeric(processed_df[col].str.replace(r'[^\d.-]', '', regex=True), errors='coerce')
                if processed_df[col].notnull().all():
                    valid_columns.append(col)
            except Exception as e:
                logger.warning(f"Skipping column {col}: {e}")

        if not valid_columns:
            st.error("No valid numeric columns were selected for balancing.")
            return pd.DataFrame(), valid_columns
        
        processed_df.dropna(subset=valid_columns, inplace=True)
        return processed_df, valid_columns
    
    def optimize_territories(self, balance_columns: List[str], date_resiliation: str = None) -> List[TerritoryMetrics]:
        """Create balanced territories considering active/inactive clients."""
        processed_df, valid_columns = self.preprocess_data(balance_columns, date_resiliation)
        if processed_df.empty:
            return []
        
        n_clients = len(processed_df)
        
        # Initialize optimization problem
        prob = LpProblem("Territory_Balancing", LpMinimize)
        
        # Decision variables
        x = LpVariable.dicts("client_assignment",
                           ((i, j) for i in range(n_clients) 
                            for j in range(self.num_territories)),
                           cat='Binary')
        
        # Calculate target values per territory
        targets = {col: processed_df[col].sum() / self.num_territories 
                  for col in valid_columns}
        
        # Territory sums for each metric
        territory_sums = {}
        for col in valid_columns:
            values = processed_df[col].values
            for j in range(self.num_territories):
                territory_sums[(col,j)] = lpSum(values[i] * x[i,j] 
                                              for i in range(n_clients))
        
        # Objective function: minimize total deviation from target
        prob += lpSum(abs(territory_sums[(col,j)] - targets[col]) 
                     for col in valid_columns 
                     for j in range(self.num_territories))
        
        # Each client in exactly one territory
        for i in range(n_clients):
            prob += lpSum(x[i,j] for j in range(self.num_territories)) == 1
            
        # Balance active and inactive clients
        active_clients = processed_df['is_active'].values
        target_active = sum(active_clients) / self.num_territories
        target_inactive = (n_clients - sum(active_clients)) / self.num_territories
        
        for j in range(self.num_territories):
            # Active clients balance
            prob += lpSum(active_clients[i] * x[i,j] for i in range(n_clients)) >= target_active - 1
            prob += lpSum(active_clients[i] * x[i,j] for i in range(n_clients)) <= target_active + 1
            
            # Inactive clients balance
            prob += lpSum((1-active_clients[i]) * x[i,j] for i in range(n_clients)) >= target_inactive - 1
            prob += lpSum((1-active_clients[i]) * x[i,j] for i in range(n_clients)) <= target_inactive + 1
        
        # Solve
        prob.solve(PULP_CBC_CMD(msg=False))
        
        # Extract results
        territories = []
        for j in range(self.num_territories):
            clients = [i for i in range(n_clients) 
                      if value(x[i,j]) > 0.5]
            
            territory_df = processed_df.iloc[clients]
            
            territories.append(TerritoryMetrics(
                territory_id=j+1,
                clients=clients,
                metrics={col: territory_df[col].sum() for col in valid_columns},
                active_clients=territory_df['is_active'].sum(),
                inactive_clients=(~territory_df['is_active']).sum()
            ))
        
        return territories

def main():
    st.set_page_config(page_title="Territory Optimizer", layout="wide")
    st.title('Territory Optimizer')
    
    uploaded_file = st.file_uploader("Upload CSV File", type=['csv'])
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file, on_bad_lines='warn', engine='python')
            st.info(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns")
            
            # Detect numeric columns and potential numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            potential_numeric_cols = df.columns[df.dtypes == 'object'].tolist()
            
            # Allow user to select numeric columns, including those that can be converted
            balance_columns = st.multiselect(
                "Select Columns for Balancing",
                options=numeric_cols + potential_numeric_cols
            )
            
            # Use a specific name for the date column
            date_resiliation = "Dt resiliation contrat all"
            
            if date_resiliation not in df.columns:
                st.error(f"The column '{date_resiliation}' does not exist in the data.")
                return
            
            num_territories = st.number_input(
                "Number of Territories",
                min_value=2,
                value=2,
                help="Enter desired number of territories"
            )
            
            if st.button("Optimize Territories"):
                if not balance_columns:
                    st.warning("Please select at least one column for balancing.")
                else:
                    optimizer = TerritoryOptimizer(df)
                    optimizer.num_territories = int(num_territories)
                    
                    territories = optimizer.optimize_territories(
                        balance_columns,
                        date_resiliation
                    )
                    
                    if territories:
                        # Display results
                        metrics_df = pd.DataFrame([
                            {
                                'Territory': t.territory_id,
                                'Active Clients': t.active_clients,
                                'Inactive Clients': t.inactive_clients,
                                **{f'{col} Sum': t.metrics[col] for col in balance_columns}
                            }
                            for t in territories
                        ])
                        
                        st.write("Territory Summary:")
                        st.dataframe(metrics_df)
                        
                        for territory in territories:
                            with st.expander(f"Territory {territory.territory_id}"):
                                territory_df = df.iloc[territory.clients]
                                st.write(f"Active Clients: {territory.active_clients}")
                                st.write(f"Inactive Clients: {territory.inactive_clients}")
                                
                                for col, value in territory.metrics.items():
                                    st.write(f"{col} Sum: {value:,.2f}")
                                    
                                st.dataframe(territory_df)
                                
                                csv = territory_df.to_csv(index=False)
                                st.download_button(
                                    f"Download Territory {territory.territory_id}",
                                    data=csv,
                                    file_name=f'territory_{territory.territory_id}.csv',
                                    mime='text/csv'
                                )
                    else:
                        st.error("Optimization failed. Please check your data and selections.")
                        
        except Exception as e:
            st.error(f"Error: {str(e)}")
            logger.exception("Error in main function")

if __name__ == "__main__":
    main()
