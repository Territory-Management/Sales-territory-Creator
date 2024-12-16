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

            # Improved column selection interface
            st.subheader("Column Selection for Territory Balancing")
            
            # Separate columns by type with descriptions
            numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
            
            # Create tabs for different column types
            col_tab1, col_tab2 = st.tabs(["Numeric Columns", "Categorical Columns"])
            
            with col_tab1:
                st.write("**Available Numeric Columns:**")
                for col in numeric_columns:
                    st.write(f"- {col}")
                    col_stats = df[col].describe()
                    st.write(f"  Min: {col_stats['min']:.2f}, Max: {col_stats['max']:.2f}, Mean: {col_stats['mean']:.2f}")
            
            with col_tab2:
                st.write("**Categorical Columns:**")
                for col in categorical_columns:
                    st.write(f"- {col}")
                    st.write(f"  Unique values: {df[col].nunique()}")

            # Enhanced column selection with column groups
            st.subheader("Configure Territory Balancing")
            
            # Create column groups for easier selection
            column_groups = {
                "Revenue Metrics": [col for col in numeric_columns if any(term in col.lower() 
                    for term in ['revenue', 'sales', 'income', 'profit'])],
                "Customer Metrics": [col for col in numeric_columns if any(term in col.lower() 
                    for term in ['customer', 'client', 'account'])],
                "Other Numeric Metrics": [col for col in numeric_columns if not any(term in col.lower() 
                    for term in ['revenue', 'sales', 'income', 'profit', 'customer', 'client', 'account'])]
            }

            # Column selection with groups
            selected_groups = st.multiselect(
                "Select metric groups to consider",
                options=list(column_groups.keys()),
                default=["Revenue Metrics"]
            )

            # Flatten selected columns from groups
            available_balance_columns = []
            for group in selected_groups:
                available_balance_columns.extend(column_groups[group])

            # Column selection with statistics
            balance_columns = []
            weights = []
            
            if available_balance_columns:
                st.write("**Configure Balance Metrics:**")
                for col in available_balance_columns:
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        include = st.checkbox(f"Include {col}", key=f"include_{col}")
                        if include:
                            balance_columns.append(col)
                    
                    with col2:
                        stats = df[col].describe()
                        st.write(f"Mean: {stats['mean']:.2f}")
                        st.write(f"Std: {stats['std']:.2f}")
                    
                    with col3:
                        if include:
                            weight = st.slider(
                                f"Weight for {col}",
                                0.0, 1.0, 1.0,
                                key=f"weight_{col}"
                            )
                            weights.append(weight)

            # Territory configuration
            st.subheader("Territory Configuration")
            col1, col2 = st.columns(2)
            
            with col1:
                num_territories = st.number_input(
                    "Number of territories",
                    min_value=2,
                    max_value=20,
                    value=2,
                    help="How many territories do you want to create?"
                )
            
            with col2:
                max_imbalance = st.slider(
                    "Maximum allowed imbalance",
                    min_value=0.01,
                    max_value=0.20,
                    value=0.05,
                    format="%.2f",
                    help="Maximum allowed difference between territories (as a ratio)"
                )

            # Preview selected metrics
            if balance_columns:
                st.subheader("Selected Metrics Preview")
                metrics_df = df[balance_columns].describe()
                st.write(metrics_df)
                
                # Show correlation matrix
                st.subheader("Correlation Matrix")
                corr_matrix = df[balance_columns].corr()
                st.write(corr_matrix)

            # Select output columns
            output_columns = st.multiselect(
                "Select columns to include in territory output",
                options=df.columns.tolist(),
                default=balance_columns + [col for col in categorical_columns if 'name' in col.lower() or 'id' in col.lower()]
            )

            if st.button("Create Balanced Territories"):
                if not balance_columns:
                    st.error("Please select at least one metric for balancing territories")
                    return

                try:
                    territories = territory_creator.create_equitable_territories(
                        num_territories,
                        balance_columns,
                        weights,
                        max_imbalance
                    )

                    # Display results in tabs
                    territory_tabs = st.tabs([f"Territory {i+1}" for i in range(num_territories)])
                    
                    for i, (tab, territory) in enumerate(zip(territory_tabs, territories)):
                        with tab:
                            st.write(territory[output_columns])
                            
                            # Summary metrics
                            st.subheader("Territory Summary")
                            summary_cols = st.columns(len(balance_columns))
                            for col, summary_col in zip(balance_columns, summary_cols):
                                with summary_col:
                                    st.metric(
                                        label=col,
                                        value=f"{territory[col].sum():,.2f}",
                                        delta=f"{(territory[col].sum() / df[col].sum() * 100):.1f}%"
                                    )
                            
                            st.download_button(
                                label=f"Download Territory {i+1}",
                                data=territory[output_columns].to_csv(index=False),
                                file_name=f"territory_{i+1}.csv",
                                mime="text/csv"
                            )

                except Exception as e:
                    st.error(f"Error creating territories: {str(e)}")

if __name__ == '__main__':
    main()
