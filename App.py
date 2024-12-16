import streamlit as st
import pandas as pd
import numpy as np
import base64
import chardet
from typing import List, Optional, Union

class TerritoryCreator:
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the TerritoryCreator with a DataFrame
        
        :param df: Input DataFrame to create territories from
        """
        self.original_df = df.copy()
        self.df = df.copy()

    def detect_encoding(self, file) -> str:
        """
        Detect the encoding of the uploaded file
        
        :param file: File object to detect encoding
        :return: Detected encoding
        """
        raw_data = file.read()
        file.seek(0)  # Reset file pointer
        result = chardet.detect(raw_data)
        return result['encoding']

    @staticmethod
    def load_data(file) -> Optional[pd.DataFrame]:
        """
        Load CSV file with multiple encoding attempts
        
        :param file: File object to load
        :return: Loaded DataFrame or None
        """
        # List of encodings to try
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'windows-1252']
        
        try:
            # Detect the encoding
            detected_encoding = chardet.detect(file.read())['encoding']
            file.seek(0)  # Reset file pointer
            encodings.insert(0, detected_encoding)  # Try detected encoding first
            
            for encoding in encodings:
                try:
                    file.seek(0)  # Reset file pointer
                    df = pd.read_csv(file, encoding=encoding, sep=None, engine='python')
                    st.success(f"Successfully loaded file using {encoding} encoding")
                    return df
                except Exception as e:
                    st.warning(f"Failed to load file with {encoding} encoding: {str(e)}")
        except Exception as e:
            st.error(f"Error detecting file encoding: {str(e)}")
        
        # If all attempts fail
        st.error("Could not load the file. Please check the file format and encoding.")
        return None

    def _normalize_balance_column(self, balance_column: str) -> None:
        """
        Normalize the balance column to ensure numeric values
        
        :param balance_column: Name of the column to normalize
        """
        if not pd.api.types.is_numeric_dtype(self.df[balance_column]):
            try:
                # Remove currency symbols, commas, and convert to float
                self.df[balance_column] = (
                    self.df[balance_column]
                    .astype(str)
                    .str.replace(r'[€$£,]', '', regex=True)
                    .str.replace(',', '.')
                    .astype(float)
                )
            except Exception as e:
                st.error(f"Could not convert {balance_column} to numeric values: {str(e)}")
                raise

    def create_equitable_territories(
        self, 
        num_territories: int, 
        balance_column: str, 
        groupby_column: Optional[str] = None,
        max_imbalance: float = 0.05
    ) -> List[pd.DataFrame]:
        """
        Create territories with balanced number of clients and total MRR
        
        :param num_territories: Number of territories to create
        :param balance_column: Column used for balancing MRR
        :param groupby_column: Optional column to group by before territory creation
        :param max_imbalance: Maximum allowable imbalance ratio (default 5%)
        :return: List of territory DataFrames
        """
        # Normalize the balance column
        self._normalize_balance_column(balance_column)
        
        # Prepare the DataFrame
        df_to_distribute = self.df.copy()
        
        # Group by column if specified
        if groupby_column:
            df_to_distribute = df_to_distribute.sort_values(balance_column, ascending=False)
        
        # Initialize territories
        territories = [pd.DataFrame() for _ in range(num_territories)]
        
        # Attempt to create balanced territories
        max_iterations = 1000
        for _ in range(max_iterations):
            # Reset territories
            territories = [pd.DataFrame() for _ in range(num_territories)]
            
            # Distribute records
            for _, row in df_to_distribute.iterrows():
                # Calculate current state of territories
                territory_stats = [
                    {
                        'mrr_sum': territory[balance_column].sum() if not territory.empty else 0,
                        'client_count': len(territory)
                    } 
                    for territory in territories
                ]
                
                # Find the territory with minimum total MRR
                min_mrr_territory = min(
                    range(num_territories), 
                    key=lambda i: territory_stats[i]['mrr_sum']
                )
                
                # Add the current record to the selected territory
                territories[min_mrr_territory] = pd.concat([
                    territories[min_mrr_territory], 
                    pd.DataFrame([row])
                ])
            
            # Check balance criteria
            mrr_values = [territory[balance_column].sum() for territory in territories]
            client_counts = [len(territory) for territory in territories]
            
            # Calculate imbalance ratios
            mrr_ratio = (max(mrr_values) - min(mrr_values)) / np.mean(mrr_values)
            client_ratio = (max(client_counts) - min(client_counts)) / np.mean(client_counts)
            
            # Check if territories are balanced
            if mrr_ratio <= max_imbalance and client_ratio <= max_imbalance:
                return territories
        
        # If we can't find a perfect balance, return the best attempt
        st.warning("Could not find perfectly balanced territories. Returning best attempt.")
        return territories

    @staticmethod
    def get_table_download_link(df: pd.DataFrame, filename: str) -> str:
        """
        Create a download link for a DataFrame
        
        :param df: DataFrame to convert to CSV
        :param filename: Filename for the download
        :return: HTML link for file download
        """
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV file</a>'
        return href

def main():
    st.title('Advanced Sales Territory Creator')

    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        # File details
        file_details = {
            "Filename": uploaded_file.name,
            "Filesize": uploaded_file.size,
            "Filetype": uploaded_file.type
        }
        st.write("File Details:", file_details)

        # Load the data
        df = TerritoryCreator.load_data(uploaded_file)
        
        if df is not None:
            # Create TerritoryCreator instance
            territory_creator = TerritoryCreator(df)
            
            # Display data preview
            st.write("Data Preview:")
            st.write(df.head())

            # Column selection
            columns = df.columns.tolist()
            
            # Suggest columns for different purposes
            mrr_columns = [col for col in columns if col.lower().startswith('mrr')]
            location_columns = ['Code Postal', 'Ville', 'Pays']
            account_columns = ['Code Tiers', 'Raison Sociale', 'Account type lib', 'Account sub type lib']
            
            # Balance column selection
            default_balance_column = 'Mrr global' if 'Mrr global' in columns else columns[0]
            balance_column = st.selectbox(
                "Select column to balance territories", 
                options=columns, 
                index=columns.index(default_balance_column) if default_balance_column in columns else 0
            )
            
            # Groupby column selection
            groupby_column = st.selectbox(
                "Select column to group by (optional)", 
                options=['None'] + columns,
                index=columns.index('Code Postal') if 'Code Postal' in columns else 0
            )
            
            # Additional configuration
            num_territories = st.number_input("Number of territories", min_value=1, value=2, step=1)
            max_imbalance = st.slider(
                "Maximum Territory Imbalance (%)", 
                min_value=0.0, 
                max_value=0.2, 
                value=0.05, 
                step=0.01, 
                format="%.2f"
            )
            
            # Column selection
            default_columns = account_columns + [balance_column] if all(col in columns for col in account_columns + [balance_column]) else columns[:min(5, len(columns))]
            selected_columns = st.multiselect(
                "Select additional columns to include", 
                options=columns, 
                default=default_columns
            )

            # Create territories button
            if st.button("Create Balanced Territories"):
                # Prepare groupby column
                groupby_column = None if groupby_column == 'None' else groupby_column
                
                # Select columns for distribution
                df_selected = df[selected_columns + ([groupby_column] if groupby_column else [])]
                
                try:
                    # Create territories
                    territories = territory_creator.create_equitable_territories(
                        num_territories, 
                        balance_column, 
                        groupby_column, 
                        max_imbalance
                    )

                    # Display territory details
                    for i, territory in enumerate(territories):
                        st.subheader(f"Territory {i+1}")
                        st.write(territory)
                        st.write(f"Total {balance_column}: {territory[balance_column].sum():.2f}")
                        st.write(f"Number of Clients: {len(territory)}")
                        st.markdown(
                            TerritoryCreator.get_table_download_link(territory, f"territory_{i+1}.csv"), 
                            unsafe_allow_html=True
                        )
                    
                    # Summary statistics
                    st.subheader("Territory Summary Statistics")
                    summary = pd.DataFrame({
                        'Territory': range(1, len(territories) + 1),
                        f'Total {balance_column}': [territory[balance_column].sum() for territory in territories],
                        'Number of Clients': [len(territory) for territory in territories]
                    })
                    st.write(summary)
                    st.markdown(
                        TerritoryCreator.get_table_download_link(summary, "territory_summary.csv"), 
                        unsafe_allow_html=True
                    )

                except Exception as e:
                    st.error(f"Error creating territories: {str(e)}")

if __name__ == '__main__':
    main()
