import streamlit as st
import pandas as pd
import numpy as np
import base64

def load_data(file):
    return pd.read_csv(file)

def create_territories(df, num_territories, balance_column, groupby_column=None):
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

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        st.write("Data Preview:")
        st.write(df.head())

        columns = df.columns.tolist()
        
        # Suggest columns for different purposes
        mrr_columns = [col for col in columns if col.lower().startswith('mrr')]
        location_columns = ['Code Postal', 'Ville', 'Pays']
        account_columns = ['Code Tiers', 'Raison Sociale', 'Account type lib', 'Account sub type lib']
        
        balance_column = st.selectbox("Select column to balance territories", 
                                      options=columns, 
                                      index=columns.index('Mrr global') if 'Mrr global' in columns else 0)
        
        groupby_column = st.selectbox("Select column to group by (optional)", 
                                      options=['None'] + columns,
                                      index=columns.index('Code Postal') if 'Code Postal' in columns else 0)
        
        selected_columns = st.multiselect("Select additional columns to include", 
                                          options=columns, 
                                          default=account_columns + [balance_column])
        
        num_territories = st.number_input("Number of territories", min_value=1, value=2, step=1)

        if st.button("Create Territories"):
            if groupby_column == 'None':
                groupby_column = None
            df_selected = df[selected_columns + ([groupby_column] if groupby_column else [])]
            territories = create_territories(df_selected, num_territories, balance_column, groupby_column)

            for i, territory in enumerate(territories):
                st.subheader(f"Territory {i+1}")
                st.write(territory)
                st.write(f"Total {balance_column}: {territory[balance_column].sum()}")
                st.markdown(get_table_download_link(territory, f"territory_{i+1}.csv"), unsafe_allow_html=True)
            
            # Summary statistics
            st.subheader("Summary Statistics")
            summary = pd.DataFrame({
                'Territory': range(1, num_territories + 1),
                f'Total {balance_column}': [territory[balance_column].sum() for territory in territories],
                'Number of Accounts': [len(territory) for territory in territories]
            })
            st.write(summary)
            st.markdown(get_table_download_link(summary, "territory_summary.csv"), unsafe_allow_html=True)

if __name__ == '__main__':
    main()
