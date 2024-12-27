import pandas as pd
import numpy as np

def clean_numeric_value(value) -> float:
    """Convert a value to a numeric type, handling strings and non-numeric characters."""
    try:
        return float(re.sub(r"[^\d.-]", "", str(value)))
    except (ValueError, TypeError):
        return 0.0

def create_balanced_territories(df, num_territories):
    """Distribute rows into territories with balanced row count and column sums."""
    # Calculate total number of rows
    total_rows = len(df)

    # Determine rows per territory and handle any remainders
    rows_per_territory = total_rows // num_territories
    extra_rows = total_rows % num_territories

    # Calculate the target sum for each column
    target_sums = df.applymap(clean_numeric_value).sum() / num_territories

    # Initialize territories
    territories = [pd.DataFrame(columns=df.columns) for _ in range(num_territories)]

    # Score function to determine best fit for a row
    def calculate_score(territory, row):
        current_sums = territory.applymap(clean_numeric_value).sum()
        projected_sums = current_sums + row.apply(clean_numeric_value)
        return np.sum((projected_sums - target_sums) ** 2)

    # Sort rows by their total value to distribute high-value rows early
    df_sorted = df.assign(_total=df.applymap(clean_numeric_value).sum(axis=1)).sort_values('_total', ascending=False).drop(columns='_total')

    # Distribute rows into territories
    for index, row in df_sorted.iterrows():
        best_territory = min(range(num_territories), key=lambda i: calculate_score(territories[i], row))
        territories[best_territory] = territories[best_territory].append(row)

    # Distribute any extra rows
    for extra_index in range(extra_rows):
        row = df_sorted.iloc[-(extra_index + 1)]
        best_territory = min(range(num_territories), key=lambda i: calculate_score(territories[i], row))
        territories[best_territory] = territories[best_territory].append(row)

    return territories

# Example usage
df = pd.read_csv('data.csv')
territories = create_balanced_territories(df, 3)
for i, territory in enumerate(territories):
    territory.to_csv(f'territory_{i+1}.csv', index=False)
