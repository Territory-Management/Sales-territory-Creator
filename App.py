import streamlit as st
import pandas as pd
import numpy as np
import chardet
from typing import List, Optional, Dict, Any, Tuple
import logging
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.stats import entropy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TerritoryMetrics:
    """Data class to store territory metrics"""
    territory_id: int
    total_clients: int
    column_totals: Dict[str, float]
    column_means: Dict[str, float]

class TerritoryBalancer:
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the Territory Balancer with input validation
        
        Args:
            df (pd.DataFrame): Input dataframe containing territory data
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        
        if df.empty:
            raise ValueError("Input DataFrame cannot be empty")
            
        self.original_df = df.copy()
        self.df = df.copy()
        self.num_territories = 2
        self.balance_columns: List[str] = []
        self.variation_threshold = 0.2  # 20% variation allowed
        
    def preprocess_data(self, balance_columns: List[str]) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Enhanced data preprocessing with error handling and validation
        
        Args:
            balance_columns (List[str]): Columns to use for balancing territories
            
        Returns:
            Tuple containing processed DataFrame and numpy array of processed columns
        """
        if not balance_columns:
            raise ValueError("At least one balance column must be provided")
            
        self.balance_columns = balance_columns
        processed_df = self.df.copy()
        
        try:
            for col in balance_columns:
                if col not in processed_df.columns:
                    raise KeyError(f"Column {col} not found in DataFrame")
                    
                # Enhanced numeric conversion with better error handling
                processed_df[col] = pd.to_numeric(
                    processed_df[col].astype(str)
                    .str.replace(r'[€$£,]', '', regex=True)
                    .str.strip()
                    .replace('', np.nan),
                    errors='coerce'
                )
            
            # Remove rows with NaN and log the count
            initial_rows = len(processed_df)
            processed_df.dropna(subset=balance_columns, inplace=True)
            rows_dropped = initial_rows - len(processed_df)
            
            if rows_dropped > 0:
                logger.warning(f"Dropped {rows_dropped} rows containing NaN values")
            
            if processed_df.empty:
                raise ValueError("All rows contained NaN values after preprocessing")
            
            # Standardize features with error handling
            scaler = StandardScaler()
            processed_columns = scaler.fit_transform(processed_df[balance_columns])
            
            return processed_df, processed_columns
            
        except Exception as e:
            logger.error(f"Error in preprocessing data: {str(e)}")
            raise
    
    def advanced_distribution_strategy(
        self, 
        processed_df: pd.DataFrame, 
        processed_columns: np.ndarray,
        max_iterations: int = 5
    ) -> List[pd.DataFrame]:
        """
        Enhanced territory distribution with multiple attempts
        
        Args:
            processed_df (pd.DataFrame): Preprocessed DataFrame
            processed_columns (np.ndarray): Processed numeric columns
            max_iterations (int): Maximum number of clustering attempts
            
        Returns:
            List of DataFrames representing territories
        """
        best_territories = None
        best_score = float('inf')
        
        for iteration in range(max_iterations):
            try:
                # K-Means Clustering with different random states
                kmeans = KMeans(
                    n_clusters=self.num_territories, 
                    random_state=42 + iteration,
                    n_init=10
                )
                cluster_labels = kmeans.fit_predict(processed_columns)
                processed_df['Cluster'] = cluster_labels
                
                territories = []
                for cluster in range(self.num_territories):
                    cluster_data = processed_df[processed_df['Cluster'] == cluster]
                    
                    # Skip empty clusters
                    if cluster_data.empty:
                        continue
                    
                    # Enhanced entropy calculation with error handling
                    try:
                        cluster_data['entropy_score'] = cluster_data[self.balance_columns].apply(
                            lambda row: entropy(np.abs(row) + 1e-10),  # Add small constant to prevent log(0)
                            axis=1
                        )
                    except Exception as e:
                        logger.warning(f"Error calculating entropy scores: {str(e)}")
                        cluster_data['entropy_score'] = 0
                    
                    sorted_cluster = cluster_data.sort_values('entropy_score', ascending=False)
                    territories.append(sorted_cluster)
                
                # Calculate distribution score
                score = self._calculate_distribution_score(territories)
                
                if score < best_score:
                    best_score = score
                    best_territories = territories
                    
            except Exception as e:
                logger.error(f"Error in distribution attempt {iteration}: {str(e)}")
                continue
        
        if best_territories is None:
            raise RuntimeError("Failed to create valid territory distribution")
            
        return best_territories
    
    def _calculate_distribution_score(self, territories: List[pd.DataFrame]) -> float:
        """
        Calculate a score representing how well balanced the territories are
        
        Args:
            territories (List[pd.DataFrame]): List of territory DataFrames
            
        Returns:
            float: Score representing territory balance (lower is better)
        """
        scores = []
        for col in self.balance_columns:
            totals = [territory[col].sum() for territory in territories]
            mean_total = np.mean(totals)
            variation = np.std(totals) / mean_total if mean_total != 0 else float('inf')
            scores.append(variation)
        
        return np.mean(scores)
    
    def get_territory_metrics(self, territories: List[pd.DataFrame]) -> List[TerritoryMetrics]:
        """
        Calculate comprehensive metrics for each territory
        
        Args:
            territories (List[pd.DataFrame]): List of territory DataFrames
            
        Returns:
            List[TerritoryMetrics]: List of territory metrics
        """
        metrics = []
        for i, territory in enumerate(territories):
            column_totals = {col: territory[col].sum() for col in self.balance_columns}
            column_means = {col: territory[col].mean() for col in self.balance_columns}
            
            metrics.append(TerritoryMetrics(
                territory_id=i + 1,
                total_clients=len(territory),
                column_totals=column_totals,
                column_means=column_means
            ))
        
        return metrics
    
    def validate_territories(self, territories: List[pd.DataFrame]) -> Tuple[bool, Dict[str, Any]]:
        """
        Enhanced territory validation with detailed feedback
        
        Args:
            territories (List[pd.DataFrame]): List of territory DataFrames
            
        Returns:
            Tuple containing validation result and detailed metrics
        """
        metrics = self.get_territory_metrics(territories)
        validation_results = {}
        
        for col in self.balance_columns:
            totals = [m.column_totals[col] for m in metrics]
            mean_total = np.mean(totals)
            max_allowed = mean_total * (1 + self.variation_threshold)
            min_allowed = mean_total * (1 - self.variation_threshold)
            
            col_valid = all(min_allowed <= total <= max_allowed for total in totals)
            validation_results[col] = {
                'valid': col_valid,
                'mean': mean_total,
                'min_allowed': min_allowed,
                'max_allowed': max_allowed,
                'actual_values': totals
            }
        
        is_valid = all(result['valid'] for result in validation_results.values())
        
        return is_valid, validation_results

def analyze_csv_structure(file_content: bytes) -> dict:
    """
    Analyze CSV file structure to identify potential issues
    """
    result = {}
    
    # Convert bytes to string for analysis
    try:
        content = file_content.decode('utf-8')
    except UnicodeDecodeError:
        try:
            content = file_content.decode('latin1')
        except Exception:
            return {"error": "Could not decode file content"}
    
    lines = content.split('\n')
    result['total_lines'] = len(lines)
    
    if len(lines) == 0:
        return {"error": "Empty file"}
    
    # Analyze first few lines
    header = lines[0].strip()
    result['possible_delimiters'] = {}
    for delimiter in [',', ';', '\t', '|']:
        result['possible_delimiters'][delimiter] = header.count(delimiter)
    
    # Check column consistency
    column_counts = []
    for i, line in enumerate(lines[:20]):  # Check first 20 lines
        if line.strip():  # Skip empty lines
            for delimiter in [',', ';', '\t', '|']:
                if delimiter in line:
                    column_counts.append((i+1, len(line.split(delimiter))))
    
    result['column_counts'] = column_counts
    
    return result

def main():
    try:
        st.set_page_config(page_title="Territory Balancer", layout="wide")
        
        st.title('Advanced Territory Balancer 🌐')
        
        uploaded_file = st.file_uploader(
            "Upload CSV File", 
            type=['csv'], 
            help="Upload your client data CSV file"
        )
        
        if uploaded_file:
            try:
                # Add CSV structure analysis
                with st.expander("CSV File Analysis", expanded=True):
                    st.info("Analyzing CSV structure...")
                    analysis_result = analyze_csv_structure(file_content)
                    
                    if "error" in analysis_result:
                        st.error(f"Analysis error: {analysis_result['error']}")
                    else:
                        st.write("File structure:")
                        st.write(f"- Total lines: {analysis_result['total_lines']}")
                        st.write("- Possible delimiters found:")
                        for delimiter, count in analysis_result['possible_delimiters'].items():
                            if count > 0:
                                st.write(f"  - '{delimiter}': {count} occurrences in header")
                        
                        # Show column count variations
                        if analysis_result['column_counts']:
                            varying_columns = len(set(count for _, count in analysis_result['column_counts'])) > 1
                            if varying_columns:
                                st.warning("⚠️ Inconsistent number of columns detected:")
                                for line_num, count in analysis_result['column_counts']:
                                    if count != analysis_result['column_counts'][0][1]:
                                        st.write(f"  - Line {line_num}: {count} columns")
                
                # First, read the file content
                file_content = uploaded_file.read()
                uploaded_file.seek(0)  # Reset file pointer
                
                # Use chardet to detect encoding
                import chardet
                detected = chardet.detect(file_content)
                if detected['confidence'] > 0.5:
                    try:
                        df = pd.read_csv(uploaded_file, encoding=detected['encoding'])
                        st.success(f"File loaded successfully with {detected['encoding']} encoding (confidence: {detected['confidence']:.2%})")
                    except Exception as e:
                        st.warning(f"Detected encoding failed, trying alternatives. Error: {str(e)}")
                        uploaded_file.seek(0)  # Reset file pointer
                        raise
                else:
                    raise ValueError("Low confidence in detected encoding")
                    
            except Exception as first_attempt:
                # If automatic detection fails, try common encodings
                uploaded_file.seek(0)  # Reset file pointer
                encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252', 'utf-16', 'ascii']
                df = None
                
                for encoding in encodings:
                    try:
                        uploaded_file.seek(0)  # Reset file pointer for each attempt
                        df = pd.read_csv(uploaded_file, encoding=encoding)
                        st.success(f"File loaded successfully with {encoding} encoding (fallback)")
                        break
                    except Exception as e:
                        st.warning(f"Failed with {encoding} encoding: {str(e)}")
                        continue
                
                if df is None:
                    st.error("Could not read the file with any supported encoding. Please ensure your CSV file is properly formatted and try saving it as UTF-8.")
                    st.info("Tips for fixing CSV encoding issues:" + 
                           "\n- Open the file in a text editor and save it as UTF-8" +
                           "\n- If using Excel, try 'Save As' and choose 'CSV UTF-8' format" +
                           "\n- Check for any special characters or formatting issues in the file")
                    return
            
            # Display data overview in an expandable section
            with st.expander("Data Overview", expanded=True):
                st.dataframe(df.head())
                st.write(f"Total rows: {len(df)}")
                st.write(f"Total columns: {len(df.columns)}")
            
            # Column selection with better UX
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_columns:
                st.warning("No numeric columns found in the data")
                return
                
            balance_columns = st.multiselect(
                "Select Columns for Territory Balancing",
                options=numeric_columns,
                help="Choose the numeric columns to use for balancing territories"
            )
            
            # Enhanced territory configuration
            col1, col2 = st.columns(2)
            with col1:
                num_territories = st.slider(
                    "Number of Territories", 
                    min_value=2, 
                    max_value=min(10, len(df)),
                    value=3,
                    help="Select the number of territories to create"
                )
            
            with col2:
                variation_tolerance = st.slider(
                    "Variation Tolerance (%)", 
                    min_value=5, 
                    max_value=50, 
                    value=20,
                    help="Maximum allowed variation between territory totals"
                )
            
            if st.button("Create Territories", help="Click to generate balanced territories"):
                if not balance_columns:
                    st.warning("Please select at least one numeric column")
                    return
                
                try:
                    with st.spinner("Creating territories..."):
                        balancer = TerritoryBalancer(df)
                        balancer.num_territories = num_territories
                        balancer.variation_threshold = variation_tolerance / 100
                        
                        processed_df, processed_columns = balancer.preprocess_data(balance_columns)
                        territories = balancer.advanced_distribution_strategy(
                            processed_df, 
                            processed_columns
                        )
                        
                        is_valid, validation_results = balancer.validate_territories(territories)
                        
                        if is_valid:
                            st.success("✅ Territories created successfully!")
                            
                            # Display territory metrics
                            metrics = balancer.get_territory_metrics(territories)
                            
                            for i, territory in enumerate(territories, 1):
                                with st.expander(f"Territory {i} Details"):
                                    st.write(f"Clients: {len(territory)}")
                                    for col in balance_columns:
                                        st.write(f"{col} Total: {territory[col].sum():,.2f}")
                                    
                                    st.dataframe(territory)
                                    
                                    # Download options
                                    csv = territory.to_csv(index=False)
                                    st.download_button(
                                        label=f"Download Territory {i} CSV",
                                        data=csv,
                                        file_name=f'territory_{i}.csv',
                                        mime='text/csv'
                                    )
                        else:
                            st.warning("Could not create sufficiently balanced territories.")
                            st.write("Validation Results:", validation_results)
                            
                except Exception as e:
                    st.error(f"Error creating territories: {str(e)}")
                    logger.exception("Territory creation error")
    
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        logger.exception("Unexpected error in main function")

if __name__ == "__main__":
    main()
