import os
import requests
import pandas as pd

file_url = "https://raw.githubusercontent.com/MPIDR/Global-flows-and-rates-of-international-migration-of-scholars/master/data_processed/scopus_2024_V1_scholarlymigration_countryflows_enriched.csv"
initial_data_path = "initial_data_files"
processed_data_path = "processed_data"
output_file = os.path.join(initial_data_path, "scopus_2024_migration.csv")

os.makedirs(initial_data_path, exist_ok=True)
os.makedirs(processed_data_path, exist_ok=True)


def download_file(url, output_path):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(output_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024):
                file.write(chunk)
        print(f"Migration file downloaded successfully: {output_path}")
    else:
        print(f"Failed to download migration file. HTTP Status Code: {response.status_code}")


def creating_adj_matrix(data, path_to_save):
    data.year = data.year.astype(int)
    countries = sorted(set(data['iso3codefrom']).union(set(data['iso3codeto'])))

    for year in range(data['year'].min(), data['year'].max() + 1):
        adj_matrix = pd.DataFrame(0, columns=countries, index=countries)

        data_year = data[data['year'] == year][['iso3codefrom', 'iso3codeto']]

        for _, row in data_year.iterrows():
            adj_matrix.loc[row['iso3codefrom'], row['iso3codeto']] = 1

        file_path = os.path.join(path_to_save, f'adj_matrix_{year}.csv')
        adj_matrix.to_csv(file_path, index=True)
        print(f"Binary adjacency matrix saved for {year}: {file_path}")


def creating_adj_matrix_2q(data, path_to_save, thresholds):
    """
    Creates adjacency matrices based on multiple quantile ranges.
    """

    data.year = data.year.astype(int)
    countries = sorted(set(data['iso3codefrom']).union(set(data['iso3codeto'])))
    threshold_dict = {}

    for year in range(1998, 2019):
        data_year = data[data['year'] == year]
        if not data_year.empty:
            threshold_dict[year] = [0]+[data_year['n_migrations'].quantile(t) for t in [0.4, 0.6, 0.9, 1]]
        else:
            threshold_dict[year] = [0, 0, 0, 0, 0]

    for year in range(data['year'].min(), data['year'].max() + 1):
        data_year = data[data['year'] == year][['n_migrations', 'iso3codefrom', 'iso3codeto']]

        for i, (lower, upper) in enumerate(thresholds):
            adj_matrix = pd.DataFrame(0, columns=countries, index=countries)

            for _, row in data_year.iterrows():
                if threshold_dict.get(year, [0, 0, 0, 0, 0])[i] < row['n_migrations'] <= threshold_dict.get(year, [0, 0, 0, 0, 0])[i + 1]:
                    adj_matrix.loc[row['iso3codefrom'], row['iso3codeto']] = 1

            file_path = os.path.join(
                path_to_save,
                f'adj_matrix_{int(lower * 100)}_{int(upper * 100)}_{year}.csv'
            )
            adj_matrix.to_csv(file_path, index=True)
            print(f"Threshold-based adjacency matrix saved for {year}: {file_path}")


if __name__ == "__main__":
    download_file(file_url, output_file)

    if os.path.exists(output_file):


        data = pd.read_csv(output_file)
        creating_adj_matrix(data, processed_data_path)
        threshold_ranges = [(0, 0.4), (0.4, 0.6), (0.6, 0.9), (0.9, 1)]
        creating_adj_matrix_2q(data, processed_data_path, threshold_ranges)

        print("Processing completed successfully!")
    else:
        print("Error: File not found after download.")



