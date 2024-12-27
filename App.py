import streamlit as st
import pandas as pd
import numpy as np

def clean_numeric_value(value):
    """Clean and convert a value to float."""
    try:
        return float(value)
    except ValueError:
        return 0.0

def distribute_territories(df, num_territories, balance_columns):
    """
    Distribute clients into territories, balancing count and total value.

    Args:
        df (pd.DataFrame): Input dataframe.
        num_territories (int): Number of territories.
        balance_columns (list): Columns to balance.

    Returns:
        pd.DataFrame: DataFrame with an added 'Territory' column.
    """
    # Clean and ensure numeric values in balance columns
    df[balance_columns] = df[balance_columns].applymap(clean_numeric_value)

    # Calculate total value for each row
    df["_total"] = df[balance_columns].sum(axis=1)

    # Sort rows randomly to distribute them more evenly
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Initialize territories
    territories = [[] for _ in range(num_territories)]

    # Calculate balancing targets
    total_clients = len(df)
    avg_clients_per_territory = total_clients / num_territories
    total_value = df["_total"].sum()
    avg_value_per_territory = total_value / num_territories

    # Initialize counters
    territory_sums = np.zeros(num_territories)
    territory_counts = np.zeros(num_territories)

    # Distribute rows to territories
    for idx, row in df.iterrows():
        scores = [
            abs(territory_counts[i] + 1 - avg_clients_per_territory) +
            abs(territory_sums[i] + row["_total"] - avg_value_per_territory)
            for i in range(num_territories)
        ]
        best_territory = np.argmin(scores)
        territories[best_territory].append(row)
        territory_counts[best_territory] += 1
        territory_sums[best_territory] += row["_total"]

    # Combine territories into a single DataFrame
    result = pd.concat(
        [pd.DataFrame(territory).assign(Territory=i + 1) for i, territory in enumerate(territories)],
        ignore_index=True
    )

    # Drop helper column
    result.drop(columns="_total", inplace=True)

    return result

def main():
    st.title("Sales Territory Balancer")

    # File uploader
    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

    if uploaded_file is not None:
        # Load data
        df = pd.read_csv(uploaded_file)

        # Display data preview
        st.subheader("Uploaded Data")
        st.write(df.head())

        # Define columns to balance
        balance_columns = st.multiselect(
            "Select columns to balance:",
            options=df.columns.tolist(),
            default=[
                "Mrr saas quadra entreprise", "Saas comptabilite", "Saas paie", 
                "Saas ga gi", "Saas gestion commerciale"
            ]
        )

        # Number of territories
        num_territories = st.number_input("Number of territories:", min_value=2, max_value=100, value=30, step=1)

        if st.button("Distribute Territories"):
            try:
                # Distribute territories
                distributed_df = distribute_territories(df, num_territories, balance_columns)

                # Display results
                st.subheader("Distributed Territories")
                st.write(distributed_df)

                # Allow download of results
                csv = distributed_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="distributed_territories.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
