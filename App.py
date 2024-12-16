import streamlit as st
import pandas as pd
import numpy as np
import base64
import chardet
from typing import List, Optional, Union

class TerritoryCreator:
    def __init__(self, df: pd.DataFrame):
        self.original_df = df.copy()
        self.df = df.copy()

    @staticmethod
    def detect_encoding(file) -> str:
        raw_data = file.read()
        file.seek(0)
        result = chardet.detect(raw_data)
        return result['encoding']

    @staticmethod
    def load_data(file) -> Optional[pd.DataFrame]:
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'windows-1252']
        try:
            detected_encoding = chardet.detect(file.read())['encoding']
            file.seek(0)
            encodings.insert(0, detected_encoding)
            
            for encoding in encodings:
                try:
                    file.seek(0)
                    df = pd.read_csv(file, encoding=encoding, sep=None, engine='python')
                    st.success(f"Successfully loaded file using {encoding} encoding")
                    return df
                except Exception as e:
                    st.warning(f"Failed to load file with {encoding} encoding: {str(e)}")
        except Exception as e:
            st.error(f"Error detecting file encoding: {str(e)}")
        
        st.error("Could not load the file. Please check the file format and encoding.")
        return None

    def _normalize_column(self, column: str) -> None:
        """
        Normalize a column to ensure it's numeric.
        Handles currency symbols, commas, and other common formatting issues.
        """
        if not pd.api.types.is_numeric_dtype(self.df[column]):
            try:
                self.df[column] = (
                    self.df[column]
                    .astype(str)
                    .str.replace(r'[€$£,]', '', regex=True)
                    .str.replace(',', '.')
                    .astype(float)
            except Exception as e:
                st.error(f"Could not convert {column} to numeric values: {str(e)}")
                raise

    def create_equitable_territories(
        self, 
        num_territories: int, 
        balance_columns: List[str], 
        weights: Optional[List[float]] = None,
        max_imbalance: float = 0.05
    ) -> List[pd.DataFrame]:
        """
        Create territories with balanced metrics across multiple columns.
        
        :param num_territories: Number of territories to create
        :param balance_columns: List of columns to balance
        :param weights: Optional list of weights for each balance column
        :param max_imbalance: Maximum allowed imbalance ratio
        :return: List of territory DataFrames
        """
        # Normalize all balance columns
        for column in balance_columns:
            self._normalize_column(column)
        
        # Apply weights if provided, otherwise use equal weights
        if weights is None:
            weights = [1.0 / len(balance_columns)] * len(balance_columns)
        
        # Validate weights
        if len(weights) != len(balance_columns):
            raise ValueError("Number of weights must match number of balance columns")
        
        # Normalize weights
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        # Create a composite score column
        composite_column = 'territory_balance_score'
        self.df[composite_column] = sum(
            self.df[col] * weight 
            for col, weight in zip(balance_columns, normalized_weights)
        )
        
        # Sort by the composite score
        df_sorted = self.df.sort_values(composite_column, ascending=False).copy()
        
        # Create initial territories
        territories = [df_sorted.iloc[i::num_territories].reset_index(drop=True) for i in range(num_territories)]
        
        # Calculate metrics
        territory_metrics = []
        for territory in territories:
            metrics = {}
            for col in balance_columns:
                metrics[f'{col}_total'] = territory[col].sum()
                metrics[f'{col}_mean'] = territory[col].mean()
            metrics['client_count'] = len(territory)
            territory_metrics.append(metrics)
        
        # Prepare summary output
        st.subheader("Territory Balance Metrics")
        summary_data = []
        for i, metrics in enumerate(territory_metrics):
            summary_data.append({
                'Territory': i + 1,
                **{k: v for k, v in metrics.items()}
            })
        summary_df = pd.DataFrame(summary_data)
        st.write(summary_df)
        
        # Calculate imbalance ratios
        imbalance_ratios = {}
        for col in balance_columns:
            total_values = [metrics[f'{col}_total'] for metrics in territory_metrics]
            mean_value = np.mean(total_values)
            ratio = (max(total_values) - min(total_values)) / mean_value
            imbalance_ratios[col] = ratio
        
        # Check overall balance
        max_observed_imbalance = max(imbalance_ratios.values())
        st.write("Imbalance Ratios:", imbalance_ratios)
        
        if max_observed_imbalance <= max_imbalance:
            st.success(f"Territories created successfully with acceptable imbalance of {max_observed_imbalance:.2%}")
        else:
            st.warning(f"Could not find perfectly balanced territories. Maximum imbalance: {max_observed_imbalance:.2%}")
        
        # Drop the temporary composite score column
        self.df.drop(columns=[composite_column], inplace=True)
        
        return territories

    @staticmethod
    def get_table_download_link(df: pd.DataFrame, filename: str) -> str:
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV file</a>'
        return href

def main():
    st.title('Advanced Multi-Metric Sales Territory Creator')

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        file_details = {
            "Filename": uploaded_file.name,
            "Filesize": uploaded_file.size,
            "Filetype": uploaded_file.type
        }
        st.write("File Details:", file_details)

        df = TerritoryCreator.load_data(uploaded_file)
        
        if df is not None:
            territory_creator = TerritoryCreator(df)
            
            st.write("Data Preview:")
            st.write(df.head())

            # Detailed column information
            st.subheader("Columns in Your Dataset")
            
            # Create columns for better layout
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Column Names:**")
                for col in df.columns:
                    st.text(col)
            
            with col2:
                st.write("**Column Types:**")
                for col in df.columns:
                    st.text(f"{col}: {df[col].dtype}")
            
            # Separate section for column selection with more details
            st.subheader("Select Columns for Territory Balancing")
            
            # Separate the different types of columns
            numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            object_columns = df.select_dtypes(include=['object']).columns.tolist()
            
            # Column selection with more context
            st.write("**Select Numeric Columns for Balancing (recommended):**")
            balance_columns = st.multiselect(
                "Choose columns to balance territories", 
                options=numeric_columns, 
                help="Select numeric columns that represent metrics you want to distribute evenly across territories"
            )
            
            # Optional: Show column statistics for selected balance columns
            if balance_columns:
                st.subheader("Selected Columns Statistics")
                column_stats = df[balance_columns].describe()
                st.write(column_stats)
            
            # Other parameters
            num_territories = st.number_input("Number of territories", min_value=1, value=2, step=1)
            max_imbalance = st.slider(
                "Maximum Territory Imbalance (%)", 
                min_value=0.0, 
                max_value=0.2, 
                value=0.05, 
                step=0.01, 
                format="%.2f"
            )
            
            # Column weights
            if balance_columns:
                weights = []
                with st.expander("Adjust Column Weights"):
                    for col in balance_columns:
                        weight = st.slider(
                            f"Weight for {col}", 
                            min_value=0.0, 
                            max_value=1.0, 
                            value=1.0 / len(balance_columns), 
                            step=0.05, 
                            format="%.2f"
                        )
                        weights.append(weight)
            else:
                weights = None
            
            # Select columns to include in final territories
            selected_columns = st.multiselect(
                "Select columns to include in territory outputs", 
                options=df.columns.tolist(), 
                default=list(df.columns[:10])  # Default to first 10 columns
            )

            if st.button("Create Balanced Territories"):
                if not balance_columns:
                    st.error("Please select at least one numeric column to balance")
                    return
                
                df_selected = df[selected_columns]
                
                try:
                    territories = territory_creator.create_equitable_territories(
                        num_territories, 
                        balance_columns, 
                        weights,
                        max_imbalance
                    )

                    for i, territory in enumerate(territories):
                        st.subheader(f"Territory {i+1}")
                        st.write(territory)
                        
                        # Display total values for balance columns
                        for col in balance_columns:
                            st.write(f"Total {col}: {territory[col].sum():.2f}")
                        
                        st.write(f"Number of Clients: {len(territory)}")
                        st.markdown(
                            TerritoryCreator.get_table_download_link(territory, f"territory_{i+1}.csv"), 
                            unsafe_allow_html=True
                        )

                except Exception as e:
                    st.error(f"Error creating territories: {str(e)}")

if __name__ == '__main__':
    main()
