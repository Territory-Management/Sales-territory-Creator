import streamlit as st
import pandas as pd
import base64
import logging
import re
from typing import List
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clean_numeric_value(value) -> float:
    """Clean numeric values, removing non-numeric characters."""
    if pd.isna(value):
        return 0.0
    try:
        clean_value = re.sub(r'[^\d.-]', '', str(value))
        return float(clean_value) if clean_value else 0.0
    except (ValueError, TypeError):
        return 0.0

def preprocess_data(df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    """Preprocess data by filtering rows based on 'dt resiliation' and separating excluded rows."""
    # Parse "dt resiliation" to datetime
    if 'Dt resiliation contrat all' in df.columns:
        df['Dt resiliation contrat all'] = pd.to_datetime(df['Dt resiliation contrat all'], errors='coerce')
        
        # Current and next year boundaries
        current_year = datetime.now().year
        excluded = df[df['Dt resiliation contrat all'].dt.year.isin([current_year, current_year + 1])]
        remaining = df[~df.index.isin(excluded.index)]
    else:
        excluded = pd.DataFrame(columns=df.columns)
        remaining = df

    return remaining, excluded

def distribute_balanced_territories(df: pd.DataFrame, num_territories: int, balance_columns: List[str]) -> List[pd.DataFrame]:
    """Distribute accounts into balanced territories."""
    territories = [pd.DataFrame(columns=df.columns) for _ in range(num_territories)]
    territory_sums = [0.0] * num_territories
    territory_counts = [0] * num_territories

    # Calculate total value for balancing columns
    df['_total'] = df.apply(lambda row: sum(clean_numeric_value(row[col]) for col in balance_columns), axis=1)

    # Sort data by total value
    df_sorted = df.sort_values('_total', ascending=False)

    for _, row in df_sorted.iterrows():
        # Assign to the territory with the smallest total value
        min_index = min(range(num_territories), key=lambda i: (territory_counts[i], territory_sums[i]))
        territories[min_index] = pd.concat([territories[min_index], pd.DataFrame([row])], ignore_index=True)
        territory_sums[min_index] += row['_total']
        territory_counts[min_index] += 1

    # Drop '_total' column
    for i in range(num_territories):
        territories[i] = territories[i].drop(columns='_total')

    return territories

def redistribute_excluded_rows(excluded: pd.DataFrame, territories: List[pd.DataFrame], balance_columns: List[str]) -> List[pd.DataFrame]:
    """Redistribute excluded rows fairly across territories."""
    # Sort excluded rows by balancing columns
    excluded['_total'] = excluded.apply(lambda row: sum(clean_numeric_value(row[col]) for col in balance_columns), axis=1)
    excluded_sorted = excluded.sort_values('_total', ascending=False)

    # Distribute excluded rows
    territory_counts = [len(territory) for territory in territories]
    for _, row in excluded_sorted.iterrows():
        min_index = min(range(len(territories)), key=lambda i: territory_counts[i])
        territories[min_index] = pd.concat([territories[min_index], pd.DataFrame([row])], ignore_index=True)
        territory_counts[min_index] += 1

    # Drop '_total' column
    for i in range(len(territories)):
        if '_total' in territories[i].columns:
            territories[i] = territories[i].drop(columns='_total')

    return territories

def main():
    st.title('Territory Distribution Tool')

    uploaded_file = st.file_uploader("Upload CSV file", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file, dtype=str)

        if df is not None:
            st.write("Data Preview:", df.head())
            balance_columns = st.multiselect(
                "Select columns to balance",
                options=[col for col in df.columns if col != 'Code client' and col != 'Dt resiliation contrat all'],
                default=[]
            )

            if not balance_columns:
                st.error("Please select at least one column to balance")
                return

            num_territories = st.number_input(
                "Number of territories",
                min_value=2,
                max_value=len(df),
                value=2
            )

            remaining, excluded = preprocess_data(df)

            if st.button("Create Territories"):
                # Create initial territories
                territories = distribute_balanced_territories(remaining, num_territories, balance_columns)

                # Redistribute excluded rows
                territories = redistribute_excluded_rows(excluded, territories, balance_columns)

                # Display results
                combined = pd.concat(territories, ignore_index=True)
                st.write("Combined Territories:", combined)
                for i, territory in enumerate(territories):
                    with st.expander(f"Territory {i+1}"):
                        st.write(territory)

if __name__ == "__main__":
    main()
