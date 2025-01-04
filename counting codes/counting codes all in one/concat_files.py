import pandas as pd

def concatenate_csv_columns_vertically_no_merge(csv_files, output_file):
    """
    Concatenates columns of multiple CSV files vertically without merging columns with the same name.

    Parameters:
    csv_files (list of str): List of paths to CSV files.
    output_file (str): Path to save the resulting CSV file.

    Returns:
    None
    """
    all_data = []

    for i, file in enumerate(csv_files):
        # Read each file and rename columns to avoid merging
        df = pd.read_csv(file)
        df.columns = [f"{col}_{i+1}" for col in df.columns]
        all_data.append(df)

    # Concatenate all DataFrames vertically
    concatenated_data = pd.concat(all_data, axis=1)

    # Save the resulting DataFrame to a new CSV file
    concatenated_data.to_csv(output_file, index=False)
    print(f"Data successfully concatenated and saved to {output_file}.")



# Example usage
csv_files = ['./real_dataset_results/Watershed_Light_U_Net.csv' , 'real_dataset_results/Connected_Comp_Light_U_Net.csv' ,'real_dataset_results/Local_Maxima_Light_U_Net.csv']
output_file = "./real_dataset_results/concat_Light_U_net.csv"
concatenate_csv_columns_vertically_no_merge(csv_files, output_file)
