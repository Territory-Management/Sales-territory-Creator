import streamlit as st
import pandas as pd
import numpy as np
import base64
import chardet

def detect_encoding(file):
    """
    Detect the encoding of the uploaded file
    """
    raw_data = file.read()
    file.seek(0)  # Reset file pointer
    result = chardet.detect(raw_data)
    return result['encoding']

def load_data(file):
    """
    Load CSV file with multiple encoding attempts
    """
    # List of encodings to try
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'windows-1252']
    
    # Detect the encoding
    detected_encoding = detect_encoding(file)
    encodings.insert(0, detected_encoding)  # Try detected encoding first
    
    for encoding in encodings:
        try:
            file.seek(0)  # Reset file pointer
            df = pd.read_csv(file, encoding=encoding, sep=None, engine='python')
            st.success(f"Successfully loaded file using {encoding} encoding")
            return df
        except Exception as e:
            st.warning(f"Failed to load file with {encoding} encoding: {str(e)}")
    
    # If all attempts fail
    st.error("Could not load the file. Please check the file format and encoding.")
    return None

def create_territories(df, num_territories, balance_column, groupby_column=None):
    if df is None or balance_column not in df.columns:
        st.error(f"Column '{balance_column}' not found in the dataframe.")
        return []

    # Handle numeric conversion if needed
    if not pd.api.types.is_numeric_dtype(df[balance_column]):
        try:
            df[balance_column] = df[balance_column].str.replace(',', '.').astype(float)
        except:
            st.error(f"Could not convert {balance_column} to numeric values.")
            return []

    if groupby_column:
        grouped = df.groupby(groupby_column)
        territories = [pd.DataFrame() for _ in range(num_territories)]
        for _, group in grouped:
            group = group.sort_values(balance_column, ascending=False)
            territory_sums = [territory[balance_column].sum() if not territory.empty else 0 for territory in territories]
            min_territory = np.argmin(territory_sums)
            territories[min_territory] = pd.concat([territories[min_territory], group])
    else:
        total = df[balance_column].sum()
        target = total / num_territories
        
        df = df.sort_values(balance_column, ascending=False)
        territories = [pd.DataFrame() for _ in range(num_territories)]
        for _, row in df.iterrows():
            territory_sums = [territory[balance_column].sum() if not territory.empty else 0 for territory in territories]
            min_territory = np.argmin(territory_sums)
            territories[min_territory] = pd.concat([territories[min_territory], pd.DataFrame([row])])
    
    return territories

def get_table_download_link(df, filename):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV file</a>'
    return href

def main():
    st.title('Sales Territory Creator')

    # Add file encoding option
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        # Additional file information
        file_details = {
            "Filename": uploaded_file.name,
            "Filesize": uploaded_file.size,
            "Filetype": uploaded_file.type
        }
        st.write("File Details:", file_details)

        # Load the data
        df = load_data(uploaded_file)
        
        if df is not None:
            st.write("Data Preview:")
            st.write(df.head())

            columns = df.columns.tolist()
            
            # Suggest columns for different purposes
            mrr_columns = [col for col in columns if col.lower().startswith('mrr')]
            location_columns = ['Code Postal', 'Ville', 'Pays']
            account_columns = ['Code Tiers', 'Raison Sociale', 'Account type lib', 'Account sub type lib']
            
            # Fallback for balance column selection
            default_balance_column = 'Mrr global' if 'Mrr global' in columns else columns[0]
            balance_column = st.selectbox("Select column to balance territories", 
                                          options=columns, 
                                          index=columns.index(default_balance_column) if default_balance_column in columns else 0)
            
            # Groupby column selection
            groupby_column = st.selectbox("Select column to group by (optional)", 
                                          options=['None'] + columns,
                                          index=columns.index('Code Postal') if 'Code Postal' in columns else 0)
            
            # Column selection
            default_columns = account_columns + [balance_column] if all(col in columns for col in account_columns + [balance_column]) else columns[:min(5, len(columns))]
            selected_columns = st.multiselect("Select additional columns to include", 
                                              options=columns, 
                                              default=default_columns)
            
            num_territories = st.number_input("Number of territories", min_value=1, value=2, step=1)

            if st.button("Create Territories"):
                if groupby_column == 'None':
                    groupby_column = None
                
                df_selected = df[selected_columns + ([groupby_column] if groupby_column else [])]
                territories = create_territories(df_selected, num_territories, balance_column, groupby_column)

                if territories:
                    for i, territory in enumerate(territories):
                        st.subheader(f"Territory {i+1}")
                        st.write(territory)
                        st.write(f"Total {balance_column}: {territory[balance_column].sum()}")
                        st.markdown(get_table_download_link(territory, f"territory_{i+1}.csv"), unsafe_allow_html=True)
                    
                    # Summary statistics
                    st.subheader("Summary Statistics")
                    summary = pd.DataFrame({
                        'Territory': range(1, len(territories) + 1),
                        f'Total {balance_column}': [territory[balance_column].sum() for territory in territories],
                        'Number of Accounts': [len(territory) for territory in territories]
                    })
                    st.write(summary)
                    st.markdown(get_table_download_link(summary, "territory_summary.csv"), unsafe_allow_html=True)

if __name__ == '__main__':
    main()
