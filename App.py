import streamlit as st
import pandas as pd
import numpy as np
import base64
import chardet
from typing import List, Optional

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
    @st.cache_data
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

    def _normalize_balance_column(self, balance_column: str) -> None:
        if not pd.api.types.is_numeric_dtype(self.df[balance_column]):
            try:
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
        max_imbalance: float = 0.05
    ) -> List[pd.DataFrame]:
        self._normalize_balance_column(balance_column)
        df_sorted = self.df.sort_values(balance_column, ascending=False).copy()
        territories = [df_sorted.iloc[i::num_territories].reset_index(drop=True) for i in range(num_territories)]
        
        mrr_values = [territory[balance_column].sum() for territory in territories]
        client_counts = [len(territory) for territory in territories]
        
        mrr_ratio = (max(mrr_values) - min(mrr_values)) / np.mean(mrr_values)
        client_ratio = (max(client_counts) - min(client_counts)) / np.mean(client_counts)
        
        if mrr_ratio <= max_imbalance and client_ratio <= max_imbalance:
            return territories
        else:
            st.warning("Could not find perfectly balanced territories. Returning current attempt.")
            return territories

    @staticmethod
    def get_table_download_link(df: pd.DataFrame, filename: str) -> str:
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV file</a>'
        return href

def main():
    st.title('Advanced Sales Territory Creator')

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

            columns = df.columns.tolist()
            mrr_columns = [col for col in columns if col.lower().startswith('mrr')]
            account_columns = ['Code Tiers', 'Raison Sociale', 'Account type lib', 'Account sub type lib']
            
            default_balance_column = 'Mrr global' if 'Mrr global' in columns else columns[0]
            balance_column = st.selectbox(
                "Select column to balance territories", 
                options=columns, 
                index=columns.index(default_balance_column) if default_balance_column in columns else 0
            )
            
            num_territories = st.number_input("Number of territories", min_value=1, value=2, step=1)
            max_imbalance = st.slider(
                "Maximum Territory Imbalance (%)", 
                min_value=0.0, 
                max_value=0.2, 
                value=0.05, 
                step=0.01, 
                format="%.2f"
            )
            
            default_columns = account_columns + [balance_column] if all(col in columns for col in account_columns + [balance_column]) else columns[:min(5, len(columns))]
            selected_columns = st.multiselect(
                "Select additional columns to include", 
                options=columns, 
                default=default_columns
            )

            if st.button("Create Balanced Territories"):
                df_selected = df[selected_columns]
                
                try:
                    territories = territory_creator.create_equitable_territories(
                        num_territories, 
                        balance_column, 
                        max_imbalance
                    )

                    for i, territory in enumerate(territories):
                        st.subheader(f"Territory {i+1}")
                        st.write(territory)
                        st.write(f"Total {balance_column}: {territory[balance_column].sum():.2f}")
                        st.write(f"Number of Clients: {len(territory)}")
                        st.markdown(
                            TerritoryCreator.get_table_download_link(territory, f"territory_{i+1}.csv"), 
                            unsafe_allow_html=True
                        )
                    
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
