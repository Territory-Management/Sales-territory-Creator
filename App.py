import streamlit as st
import pandas as pd
import numpy as np
import base64
import logging
from typing import List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
DEFAULT_ENCODING = "utf-8"
BACKUP_ENCODING = "latin1"
MONETARY_SYMBOLS_REGEX = r'[\u20ac$\u00a3,]'

def load_data(file) -> Optional[pd.DataFrame]:
    """Load CSV file with improved handling for client code column and detailed error logging."""
    try:
        # Try to read first few lines to detect the separator and structure
        file.seek(0)
        sample_data = file.read(1024).decode(DEFAULT_ENCODING)
        logging.info(f"Sample of file content: {sample_data[:200]}")
        
        # Reset file pointer
        file.seek(0)
        
        # Try to detect separator
        import csv
        dialect = csv.Sniffer().sniff(sample_data)
        separator = dialect.delimiter
        logging.info(f"Detected separator: '{separator}'")
        
        # First attempt with detected separator
        df = pd.read_csv(
            file, 
            encoding=DEFAULT_ENCODING,
            sep=separator,
            dtype=str  # Read all columns as string initially
        )
        
        logging.info(f"Columns detected: {df.columns.tolist()}")
        
        # Check if we need to handle the first column
        if len(df.columns) > 0:
            first_col = df.columns[0]
            if first_col.startswith('Unnamed:') or 'Code client' not in df.columns:
                # Try reading the file again with the first column properly handled
                file.seek(0)
                df = pd.read_csv(
                    file,
                    encoding=DEFAULT_ENCODING,
                    sep=separator,
                    dtype=str
                )
                # Rename first column if necessary
                df.rename(columns={first_col: 'Code client'}, inplace=True)
        
        logging.info("File successfully loaded with utf-8 encoding.")
        logging.info(f"Final columns: {df.columns.tolist()}")
    except UnicodeDecodeError:
        try:
            file.seek(0)
            # Backup attempt with latin1 encoding
            df = pd.read_csv(
                file,
                encoding=BACKUP_ENCODING,
                dtype={'Code client': str}
            )
            logging.info("File successfully loaded with latin1 encoding.")
        except Exception as e:
            logging.error(f"Error loading file with latin1: {str(e)}")
            st.error("Error loading file. Please check the format.")
            return None
    except Exception as e:
        logging.error(f"Error loading file: {str(e)}")
        st.error("Error loading file. Please check the format.")
        return None

    # Ensure client code is properly formatted
    if 'Code client' in df.columns:
        df['Code client'] = df['Code client'].astype(str).str.zfill(8)
    
    # Verify minimum columns requirement
    if len(df.columns) < 2:
        st.error("The uploaded file does not have enough columns.")
        return None
        
    return df

def normalize_numeric_column(series: pd.Series) -> pd.Series:
    """Normalize columns with monetary symbols and non-numeric characters."""
    return pd.to_numeric(
        series.astype(str)
        .str.replace(MONETARY_SYMBOLS_REGEX, '', regex=True)
        .str.replace(',', '.')
        .str.strip(),
        errors='coerce'
    )

def distribute_equally(df: pd.DataFrame, num_territories: int, balance_columns: List[str]) -> List[pd.DataFrame]:
    """Distribute rows equitably by count and sum of specified columns across territories."""
    territories = [pd.DataFrame(columns=df.columns) for _ in range(num_territories)]
    
    # Ensure numerical values for sorting
    sort_values = df[balance_columns].apply(pd.to_numeric, errors='coerce')
    df_sorted = df.iloc[(-sort_values.sum(axis=1)).argsort()]

    # Initial allocation by row count and sum balancing
    sums = [0] * num_territories
    counts = [0] * num_territories
    for _, row in df_sorted.iterrows():
        min_index = min(range(num_territories), key=lambda i: (counts[i], sums[i]))
        territories[min_index] = pd.concat([territories[min_index], pd.DataFrame([row])], ignore_index=True)
        counts[min_index] += 1
        sums[min_index] += sum(normalize_numeric_column(row[col]) for col in balance_columns)

    return territories

def distribute_termination_clients_by_year(
    territories: List[pd.DataFrame], termination_clients: pd.DataFrame, years: List[int]
) -> List[pd.DataFrame]:
    """Distribute termination clients equitably among territories for each year."""
    for year in years:
        year_clients = termination_clients[
            termination_clients['Dt resiliation contrat all'].dt.year == year
        ]
        for i, client in enumerate(year_clients.itertuples(index=False)):
            territories[i % len(territories)] = pd.concat(
                [territories[i % len(territories)], pd.DataFrame([client._asdict()])],
                ignore_index=True
            )
    return territories

def create_territories(
    df: pd.DataFrame,
    num_territories: int,
    balance_columns: List[str],
    years: Optional[List[int]] = None
) -> tuple[List[pd.DataFrame], pd.DataFrame]:
    """Create balanced territories with adjustments for disparities."""
    working_df = df.copy()
    termination_clients = pd.DataFrame()

    if years:
        working_df['Dt resiliation contrat all'] = pd.to_datetime(
            working_df['Dt resiliation contrat all'], errors='coerce'
        )
        termination_clients = working_df[
            working_df['Dt resiliation contrat all'].dt.year.isin(years)
        ]
        working_df = working_df[
            ~working_df['Dt resiliation contrat all'].dt.year.isin(years)
        ]

    # Normalize balance columns
    for col in balance_columns:
        working_df[col] = normalize_numeric_column(working_df[col])

    territories = distribute_equally(working_df, num_territories, balance_columns)

    if not termination_clients.empty:
        territories = distribute_termination_clients_by_year(
            territories, termination_clients, years
        )

    for i, territory in enumerate(territories):
        territory['Territory'] = i + 1

    # Calculate metrics for each territory
    metrics = []
    for i, territory in enumerate(territories):
        metric = {'Territory': i + 1, 'Count': len(territory)}
        for col in balance_columns:
            metric[f'{col}_total'] = territory[col].sum()
        metrics.append(metric)

    return territories, pd.DataFrame(metrics)

def get_download_link(df: pd.DataFrame, filename: str) -> str:
    """Generate a CSV download link."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download {filename}</a>'

def main():
    st.title('EquiTerritory')

    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
    if uploaded_file:
        df = load_data(uploaded_file)

        if df is not None:
            st.write("Data Preview:", df.head())
            st.write("Column names detected:", df.columns.tolist())

            balance_columns = st.multiselect(
                "Select columns to balance",
                options=[col for col in df.columns if col != 'Code client'],
                default=[col for col in df.columns if col != 'Code client'][:1]
            )

            if not balance_columns:
                st.error("Please select at least one column to balance")
                return

            col1, col2 = st.columns(2)
            with col1:
                num_territories = st.number_input(
                    "Number of territories", min_value=2, max_value=len(df), value=2
                )

            years = st.multiselect(
                "Select termination years",
                options=[2024, 2025, 2026],
                default=[2024]
            )

            if st.button("Create Territories"):
                with st.spinner("Creating territories..."):
                    territories, metrics = create_territories(
                        df, num_territories, balance_columns, years
                    )

                st.subheader("Territory Metrics")
                st.write(metrics)

                combined_territories = pd.concat(territories, ignore_index=True)

                st.markdown(
                    get_download_link(combined_territories, "territory_distribution.csv"),
                    unsafe_allow_html=True
                )

                for i, territory in enumerate(territories):
                    with st.expander(f"Territory {i+1} ({len(territory)} accounts)"):
                        st.write(territory)
                        st.markdown(
                            get_download_link(territory, f"territory_{i+1}.csv"),
                            unsafe_allow_html=True
                        )

if __name__ == "__main__":
    main()
