import streamlit as st
import pandas as pd
import base64
import logging
import re
from typing import List, Optional, Dict
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clean_numeric_value(value) -> float:
    """Clean and convert a value to float."""
    if pd.isna(value):
        return 0.0
    try:
        str_value = str(value)
        clean_value = re.sub(r"[^\d.-]", "", str_value)
        return float(clean_value) if clean_value else 0.0
    except (ValueError, TypeError):
        return 0.0

def load_data(file) -> Optional[pd.DataFrame]:
    """Load CSV data and preprocess it."""
    try:
        file.seek(0)
        sample_data = file.read(1024).decode("utf-8")
        import csv
        dialect = csv.Sniffer().sniff(sample_data)
        separator = dialect.delimiter

        file.seek(0)
        df = pd.read_csv(file, sep=separator, dtype=str, encoding="utf-8")

        if df.columns[0].startswith("Unnamed:"):
            df.rename(columns={df.columns[0]: "Code client"}, inplace=True)

        if "Dt resiliation contrat all" in df.columns:
            df["Dt resiliation contrat all"] = pd.to_datetime(df["Dt resiliation contrat all"], errors="coerce")

        return df
    except Exception as e:
        logging.error(f"Error loading file: {str(e)}")
        st.error(f"Error loading file: {str(e)}")
        return None

def filter_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Filter rows based on 'Dt resiliation contrat all' date."""
    current_year = datetime.now().year
    next_year = current_year + 1
    if "Dt resiliation contrat all" in df.columns:
        return df[(df["Dt resiliation contrat all"].dt.year > next_year) | df["Dt resiliation contrat all"].isna()]
    return df

def distribute_territories(df: pd.DataFrame, num_territories: int, balance_columns: List[str]) -> List[pd.DataFrame]:
    """
    Distribute clients into territories, balancing both count and total values using a weighted approach
    and greedy assignment with normalization.
    """
    territories = [pd.DataFrame(columns=df.columns) for _ in range(num_territories)]
    
    # Calculate total values for balancing columns and normalize them
    df = df.copy()
    
    # Calculate weighted total based on all balance columns
    df["_total"] = 0
    for col in balance_columns:
        df[f"_normalized_{col}"] = df[col].apply(clean_numeric_value)
        max_val = df[f"_normalized_{col}"].max()
        if max_val > 0:
            df[f"_normalized_{col}"] = df[f"_normalized_{col}"] / max_val
        df["_total"] += df[f"_normalized_{col}"]
    
    # Sort by total value to handle largest accounts first
    df_sorted = df.sample(frac=1, random_state=42).sort_values("_total", ascending=False)
    
    # Initialize tracking variables
    territory_metrics = {i: {
        "count": 0,
        "total": 0.0,
        "column_totals": {col: 0.0 for col in balance_columns}
    } for i in range(num_territories)}
    
    target_count = len(df) / num_territories
    
    def calculate_territory_score(territory_idx: int, row: pd.Series) -> float:
        """
        Calculate a score for assigning a row to a territory based on multiple factors.
        Lower score is better.
        """
        metrics = territory_metrics[territory_idx]
        
        # Count balance factor (how far from target count)
        count_factor = metrics["count"] / target_count if target_count > 0 else 0
        
        # Value balance factors for each column
        value_factors = []
        for col in balance_columns:
            territory_total = metrics["column_totals"][col]
            row_value = clean_numeric_value(row[col])
            if row_value > 0:
                value_factors.append(territory_total / row_value)
        
        avg_value_factor = sum(value_factors) / len(value_factors) if value_factors else 1
        
        # Weights for different factors
        COUNT_WEIGHT = 0.4
        VALUE_WEIGHT = 0.6
        
        return (count_factor * COUNT_WEIGHT) + (avg_value_factor * VALUE_WEIGHT)
    
    # Assign rows to territories
    for _, row in df_sorted.iterrows():
        # Find the territory with the lowest combined score
        best_territory = min(range(num_territories),
                           key=lambda i: calculate_territory_score(i, row))
        
        # Assign to the selected territory
        territories[best_territory] = pd.concat([territories[best_territory], 
                                              pd.DataFrame([row])], 
                                              ignore_index=True)
        
        # Update territory metrics
        territory_metrics[best_territory]["count"] += 1
        territory_metrics[best_territory]["total"] += row["_total"]
        for col in balance_columns:
            territory_metrics[best_territory]["column_totals"][col] += clean_numeric_value(row[col])
    
    # Clean up temporary columns and add territory numbers
    normalized_cols = ["_total"] + [f"_normalized_{col}" for col in balance_columns]
    for i in range(num_territories):
        territories[i] = territories[i].drop(columns=normalized_cols)
        territories[i].insert(0, "Territory", i + 1)
        
        # Log distribution metrics
        logging_msg = f"Territory {i+1} - Count: {territory_metrics[i]['count']}"
        for col in balance_columns:
            logging_msg += f", {col} Total: {territory_metrics[i]['column_totals'][col]:.2f}"
        logging.info(logging_msg)
    
    return territories

def get_territory_metrics(territories: List[pd.DataFrame], balance_columns: List[str]) -> pd.DataFrame:
    """Calculate detailed metrics for each territory including variance statistics."""
    metrics = []
    
    for i, territory in enumerate(territories):
        metric = {
            "Territory": i + 1,
            "Count": len(territory)
        }
        
        # Calculate totals, averages, and standard deviations for each balance column
        for col in balance_columns:
            values = territory[col].apply(clean_numeric_value)
            total = values.sum()
            avg = total / len(territory) if len(territory) > 0 else 0
            std = values.std() if len(territory) > 0 else 0
            
            metric.update({
                f"{col}_total": total,
                f"{col}_avg": avg,
                f"{col}_std": std,
                f"{col}_min": values.min() if len(territory) > 0 else 0,
                f"{col}_max": values.max() if len(territory) > 0 else 0
            })
        
        metrics.append(metric)
    
    metrics_df = pd.DataFrame(metrics)
    
    # Add summary statistics for the entire distribution
    summary_row = {"Territory": "Summary"}
    for col in metrics_df.columns:
        if col != "Territory":
            summary_row[col] = metrics_df[col].mean()
    
    metrics_df = pd.concat([metrics_df, pd.DataFrame([summary_row])], ignore_index=True)
    
    return metrics_df

def get_download_link(df: pd.DataFrame, filename: str) -> str:
    """Generate a CSV download link for a DataFrame."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download {filename}</a>'

def main():
    st.title("Territory Distribution Tool")
    
    st.sidebar.header("Configuration")
    # Add configuration options in sidebar
    balance_weights = st.sidebar.slider(
        "Balance Weight (Higher values favor value balance over count balance)",
        min_value=0.0,
        max_value=1.0,
        value=0.6,
        step=0.1
    )

    uploaded_file = st.file_uploader("Upload CSV file", type="csv")
    if uploaded_file:
        df = load_data(uploaded_file)

        if df is not None:
            st.write("Data Preview:", df.head())
            st.write("Column names:", df.columns.tolist())

            # Filter rows based on resiliation date
            df_filtered = filter_rows(df)
            st.write(f"Filtered Data Preview ({len(df_filtered)} rows):", df_filtered.head())

            # Select balance columns
            available_columns = [col for col in df_filtered.columns 
                               if col not in ["Code client", "Dt resiliation contrat all", "Territory"]]
            
            balance_columns = st.multiselect(
                "Select columns to balance",
                options=available_columns,
                default=available_columns[:1] if available_columns else []
            )

            if not balance_columns:
                st.error("Please select at least one column to balance")
                return

            # Number of territories
            num_territories = st.number_input(
                "Number of territories",
                min_value=2,
                max_value=len(df_filtered),
                value=min(4, len(df_filtered))
            )

            if st.button("Create Territories"):
                with st.spinner("Distributing territories..."):
                    territories = distribute_territories(df_filtered, num_territories, balance_columns)
                    metrics = get_territory_metrics(territories, balance_columns)

                    # Display metrics
                    st.subheader("Territory Metrics")
                    st.dataframe(metrics)

                    # Show distribution visualizations
                    for col in balance_columns:
                        st.subheader(f"Distribution of {col}")
                        territory_totals = {
                            f"Territory {i+1}": territory[col].apply(clean_numeric_value).sum()
                            for i, territory in enumerate(territories)
                        }
                        chart_data = pd.DataFrame({
                            'Territory': list(territory_totals.keys()),
                            'Total': list(territory_totals.values())
                        })
                        chart_data.set_index('Territory', inplace=True)
                        st.bar_chart(chart_data)

                    # Combined territories download
                    combined = pd.concat(territories, ignore_index=True)
                    st.markdown(get_download_link(combined, "all_territories.csv"), 
                              unsafe_allow_html=True)

                    # Individual territory downloads
                    for i, territory in enumerate(territories):
                        with st.expander(f"Territory {i+1} ({len(territory)} accounts)"):
                            st.dataframe(territory)
                            st.markdown(
                                get_download_link(territory, f"territory_{i+1}.csv"),
                                unsafe_allow_html=True
                            )

                    # Calculate and display distribution quality metrics
                    quality_metrics = {}
                    for col in balance_columns:
                        totals = [territory[col].apply(clean_numeric_value).sum() 
                                for territory in territories]
                        quality_metrics[col] = {
                            'Coefficient of Variation': pd.Series(totals).std() / pd.Series(totals).mean()
                        }
                    
                    st.subheader("Distribution Quality Metrics")
                    st.dataframe(pd.DataFrame(quality_metrics))

if __name__ == "__main__":
    main()
