import pandas as pd
import numpy as np

def create_balanced_territories(df, num_territories):
    # Calculate the total number of rows
    total_rows = len(df)

    # Determine the number of rows per territory and handle remainders
    rows_per_territory = total_rows // num_territories
    extra_rows = total_rows % num_territories

    # Calculate the target sum for each column
    target_sums = df.sum() / num_territories

    # Initialize territories
    territories = [pd.DataFrame(columns=df.columns) for _ in range(num_territories)]

    # Create a score function to determine best fit for a row
    def calculate_score(territory, row, target_sums):
        current_sums = territory.sum()
        projected_sums = current_sums + row
        return np.sum((projected_sums - target_sums) ** 2)

    # Sort rows by their total value to distribute high-value rows early
    df_sorted = df.assign(_total=df.sum(axis=1)).sort_values('_total', ascending=False).drop(columns='_total')

    # Distribute rows into territories
    for index, row in df_sorted.iterrows():
        best_territory = min(range(num_territories), key=lambda i: calculate_score(territories[i], row, target_sums))
        territories[best_territory] = territories[best_territory].append(row)

    # Distribute extra rows if any
    for extra_index in range(extra_rows):
        row = df_sorted.iloc[-(extra_index + 1)]
        best_territory = min(range(num_territories), key=lambda i: calculate_score(territories[i], row, target_sums))
        territories[best_territory] = territories[best_territory].append(row)

    return territories

# Example usage
df = pd.read_csv('data.csv')
territories = create_balanced_territories(df, 3)
for i, territory in enumerate(territories):
    territory.to_csv(f'territory_{i+1}.csv', index=False)
