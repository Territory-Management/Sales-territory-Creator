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
    Distribute clients into territories using a bucket-based approach with strict balancing.
    """
    territories = [pd.DataFrame(columns=df.columns) for _ in range(num_territories)]
    df = df.copy()
    
    # Step 1: Clean and prepare the data
    for col in balance_columns:
        df[f"_clean_{col}"] = df[col].apply(clean_numeric_value)
    
    # Step 2: Create value buckets for better distribution
    num_buckets = 5  # Number of value ranges to consider
    for col in balance_columns:
        df[f"_bucket_{col}"] = pd.qcut(df[f"_clean_{col}"], 
                                      q=num_buckets, 
                                      labels=False, 
                                      duplicates='drop')
    
    # Calculate target metrics
    total_rows = len(df)
    target_count_per_territory = total_rows / num_territories
    
    # Initialize territory metrics
    territory_metrics = {i: {
        "count": 0,
        "values": {col: 0.0 for col in balance_columns},
        "buckets": {col: {i: 0 for i in range(num_buckets)} for col in balance_columns}
    } for i in range(num_territories)}
    
    def get_assignment_score(territory_idx: int, row: pd.Series) -> float:
        """
        Calculate assignment score with strict balancing constraints.
        Lower score is better.
        """
        metrics = territory_metrics[territory_idx]
        
        # Count balance score - how far from target count
        count_ratio = metrics["count"] / target_count_per_territory
        count_score = abs(1 - count_ratio) * 2  # Doubled importance for count
        
        # Value distribution score
        value_scores = []
        for col in balance_columns:
            current_total = metrics["values"][col]
            row_value = row[f"_clean_{col}"]
            
            # Calculate average value per territory
            avg_value = df[f"_clean_{col}"].sum() / num_territories
            
            # How far from average after adding this row?
            new_ratio = (current_total + row_value) / avg_value
            value_scores.append(abs(1 - new_ratio))
        
        value_score = sum(value_scores) / len(value_scores)
        
        # Bucket balance score
        bucket_scores = []
        for col in balance_columns:
            bucket = row[f"_bucket_{col}"]
            target_bucket_count = len(df[df[f"_bucket_{col}"] == bucket]) / num_territories
            current_bucket_count = metrics["buckets"][col][bucket]
            if target_bucket_count > 0:
                bucket_ratio = (current_bucket_count + 1) / target_bucket_count
                bucket_scores.append(abs(1 - bucket_ratio))
        
        bucket_score = sum(bucket_scores) / len(bucket_scores) if bucket_scores else 0
        
        # Combine scores with weights
        COUNT_WEIGHT = 0.4
        VALUE_WEIGHT = 0.4
        BUCKET_WEIGHT = 0.2
        
        # Add penalties for exceeding thresholds
        if count_ratio > 1.1:  # More than 10% over target count
            return float('inf')
        
        return (count_score * COUNT_WEIGHT + 
                value_score * VALUE_WEIGHT + 
                bucket_score * BUCKET_WEIGHT)
    
    # Step 3: Sort and distribute accounts
    # First sort by the highest value column
    main_value_col = max(balance_columns, 
                        key=lambda col: df[f"_clean_{col}"].sum())
    df_sorted = df.sort_values(f"_clean_{main_value_col}", ascending=False)
    
    # Distribute accounts
    for _, row in df_sorted.iterrows():
        valid_territories = [i for i in range(num_territories)
                           if territory_metrics[i]["count"] < target_count_per_territory * 1.1]
        
        if not valid_territories:
            valid_territories = range(num_territories)
        
        # Find best territory
        best_territory = min(valid_territories,
                           key=lambda i: get_assignment_score(i, row))
        
        # Assign to territory
        territories[best_territory] = pd.concat([territories[best_territory], 
                                              pd.DataFrame([row])], 
                                              ignore_index=True)
        
        # Update metrics
        territory_metrics[best_territory]["count"] += 1
        for col in balance_columns:
            value = row[f"_clean_{col}"]
            territory_metrics[best_territory]["values"][col] += value
            bucket = row[f"_bucket_{col}"]
            territory_metrics[best_territory]["buckets"][col][bucket] += 1
    
    # Clean up temporary columns
    temp_cols = ([f"_clean_{col}" for col in balance_columns] + 
                 [f"_bucket_{col}" for col in balance_columns])
    
    for i in range(num_territories):
        territories[i] = territories[i].drop(columns=temp_cols, errors='ignore')
        territories[i].insert(0, "Territory", i + 1)
        
        # Log distribution metrics
        logging_msg = f"Territory {i+1} - Count: {territory_metrics[i]['count']}"
        for col in balance_columns:
            logging_msg += f", {col}: {territory_metrics[i]['values'][col]:,.2f}"
        logging.info(logging_msg)
    
    # Validation step - print distribution quality metrics
    for col in balance_columns:
        totals = [t["values"][col] for t in territory_metrics.values()]
        counts = [t["count"] for t in territory_metrics.values()]
        logging.info(f"\nDistribution metrics for {col}:")
        logging.info(f"Count variation: {max(counts) - min(counts)}")
        logging.info(f"Value ratio: {max(totals) / min(totals) if min(totals) > 0 else 'inf':.2f}")
    
    return territories

def get_territory_metrics(territories: List[pd.DataFrame], balance_columns: List[str]) -> pd.DataFrame:
    """Calculate detailed metrics for each territory including variance statistics."""
    metrics = []
    
    for i, territory in enumerate(territories):
        metric = {
            "Territory": i + 1,
            "Count": len(territory)
        }
        
        # Calculate totals and averages for each balance column
        for col in balance_columns:
            values = territory[col].apply(clean_numeric_value)
            total = values.sum()
            avg = total / len(territory) if len(territory) > 0 else 0
            
            metric.update({
                f"{col}_total": total,
                f"{col}_avg": avg,
                f"{col}_min": values.min() if len(territory) > 0 else 0,
                f"{col}_max": values.max() if len(territory) > 0 else 0
            })
        
        metrics.append(metric)
    
    metrics_df = pd.DataFrame(metrics)
    
    # Add coefficient of variation for each metric
    for col in balance_columns:
        totals = metrics_df[f"{col}_total"]
        cv = totals.std() / totals.mean() if totals.mean() != 0 else float('inf')
        logging.info(f"Coefficient of Variation for {col}: {cv:.3f}")
    
    return metrics_df
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
