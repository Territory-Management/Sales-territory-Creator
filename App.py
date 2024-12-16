import streamlit as st
import pandas as pd
import numpy as np
import base64
import chardet
from typing import List, Optional, Union

def load_data(file) -> Optional[pd.DataFrame]:
    try:
        # Detect encoding
        raw_data = file.read()
        file.seek(0)
        encoding = chardet.detect(raw_data)['encoding']
        
        # Try to read with detected encoding
        df = pd.read_csv(file, encoding=encoding)
        return df
    except Exception as e:
        # Fallback to common encodings
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'windows-1252']
        for enc in encodings:
            try:
                file.seek(0)
                df = pd.read_csv(file, encoding=enc)
                return df
            except:
                continue
        st.error(f"Error loading file: {str(e)}")
        return None

def normalize_numeric_column(series: pd.Series) -> pd.Series:
    try:
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
    # Create working copy
    working_df = df.copy()
    
    # Normalize columns
    for col in balance_columns:
        working_df[col] = normalize_numeric_column(working_df[col])
    
    # Calculate weighted score
    if weights is None:
        weights = [1/len(balance_columns)] * len(balance_columns)
    
    working_df['score'] = sum(
        working_df[col] * weight 
        for col, weight in zip(balance_columns, weights)
    )
    
    # Sort and distribute
    working_df = working_df.sort_values('score', ascending=False)
    territories = [working_df.iloc[i::num_territories] for i in range(num_territories)]
    
    # Calculate metrics
    metrics = []
    for i, territory in enumerate(territories):
        metric = {'Territory': i + 1, 'Count': len(territory)}
        for col in balance_columns:
            metric[f'{col}_total'] = territory[col].sum()
        metrics.append(metric)
    
    return territories, pd.DataFrame(metrics)

def get_download_link(df: pd.DataFrame, filename: str) -> str:
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download {filename}</a>'

def main():
    st.title('Fast Territory Balancer')

    uploaded_file = st.file_uploader("Upload CSV file", type="csv")
    if uploaded_file:
        df = load_data(uploaded_file)
        
        if df is not None:
            st.write("Data Preview:", df.head())

            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            balance_columns = st.multiselect(
                "Select columns to balance",
                options=numeric_cols,
                default=numeric_cols[:1]
            )

            col1, col2 = st.columns(2)
            with col1:
                num_territories = st.number_input(
                    "Number of territories",
                    min_value=2,
                    max_value=len(df),
                    value=2
                )
            
            with col2:
                use_weights = st.checkbox("Use custom weights", value=False)

            weights = None
            if use_weights and balance_columns:
                weights = []
                for col in balance_columns:
                    weight = st.slider(
                        f"Weight for {col}",
                        0.0, 1.0, 1.0/len(balance_columns),
                        0.1
                    )
                    weights.append(weight)
                
                # Normalize weights
                total = sum(weights)
                weights = [w/total for w in weights]

            if st.button("Create Territories"):
                if not balance_columns:
                    st.error("Please select at least one column to balance")
                    return

                territories, metrics = create_territories(
                    df, 
                    num_territories, 
                    balance_columns,
                    weights
                )

                st.subheader("Territory Metrics")
                st.write(metrics)

                for i, territory in enumerate(territories):
                    with st.expander(f"Territory {i+1} ({len(territory)} accounts)"):
                        st.write(territory)
                        st.markdown(
                            get_download_link(territory, f"territory_{i+1}.csv"),
                            unsafe_allow_html=True
                        )

if __name__ == "__main__":
    main()
