from pathlib import Path
from geopy.distance import geodesic
import numpy as np
import requests
import pandas as pd


def create_geo_file(path_to_save):
    try:
        base_path = Path("initial_data_files")
        ISO2_ISO3 = pd.read_csv(base_path / "ISO2_ISO3.csv", delimiter=';')
        geo_countries = pd.read_csv(base_path / "countries.csv", delimiter=',')
        geo_countries = geo_countries.merge(ISO2_ISO3, how='left', right_on='ISO2', left_on='ISO')
        canarias_idx = geo_countries[geo_countries['COUNTRY'] == 'Canarias'].index
        if not canarias_idx.empty:
            geo_countries.drop(index=canarias_idx, inplace=True)
        geo_countries.drop(['COUNTRY', 'ISO', 'COUNTRYAFF', 'AFF_ISO', 'ISO2'], axis=1, inplace=True)
        for i in ['PYF', 'WSM', 'TON', 'ASM']:
            idx = geo_countries[geo_countries['ISO3'] == i].index
            if not idx.empty:
                geo_countries.loc[idx, 'longitude'] *= -1
                geo_countries.loc[idx, 'latitude'] *= -1
        new_rows = pd.DataFrame({'longitude': [114.1694, 113.5461],
                                 'latitude': [22.3193, 22.2006],
                                 'ISO3': ['HKG', 'MAC']})
        geo_countries = pd.concat([geo_countries, new_rows], ignore_index=True)
        grl_idx = geo_countries[geo_countries['ISO3'] == 'GRL'].index
        if not grl_idx.empty:
            geo_countries.loc[grl_idx, 'latitude'] = 66
        output_path = path_to_save / "processed_geo_countries.csv"
        geo_countries.to_csv(output_path, index=False)
        print(f"Geo file successfully created: {output_path}")
        return geo_countries
    except Exception as e:
        print(f"Error creating geo file: {e}")
        return None


def blueprint_matrix(data):
    data.year = data.year.astype(int)
    countries = sorted(set(data['iso3codefrom']).union(data['iso3codeto']))
    return pd.DataFrame(index=countries, columns=countries)

def haversine_distance(lat1, lon1, lat2, lon2):
    return geodesic((lat1, lon1), (lat2, lon2)).kilometers


def distance_matrix(geo_countries, data, path_to_save):
    try:
        path_to_save = Path(path_to_save)
        adj_matrix = blueprint_matrix(data)
        countries = adj_matrix.columns

        for i, country1 in enumerate(countries):
            lat1, lon1 = geo_countries.loc[geo_countries['ISO3'] == country1, ['latitude', 'longitude']].values[0]
            for j, country2 in enumerate(countries):
                if i <= j:
                    lat2, lon2 = geo_countries.loc[geo_countries['ISO3'] == country2, ['latitude', 'longitude']].values[
                        0]
                    distance = haversine_distance(lat1, lon1, lat2, lon2)
                    adj_matrix.loc[country1, country2] = distance
                    adj_matrix.loc[country2, country1] = distance

        adj_matrix = adj_matrix.apply(pd.to_numeric)
        adj_matrix = np.log(adj_matrix.replace(0, np.nan))
        adj_matrix = adj_matrix.fillna(-1 / np.sqrt(np.finfo(float).eps))
        output_path = path_to_save / "log_distance.csv"
        adj_matrix.to_csv(output_path, index=True)
        print(f"Log distance matrix successfully created: {output_path}")
    except Exception as e:
        print(f"Error creating distance file: {e}")
        return None


def language_matrix(data, path_to_save):
    try:
        path_to_save = Path(path_to_save)
        file_path = Path("initial_data_files") / "language_data.csv"
        url = "https://www.usitc.gov/data/gravity/dicl_database.csv"
        response = requests.get(url)
        with open(file_path, "wb") as file:
            file.write(response.content)

        data_lang = pd.read_csv(file_path)
        adj_matrix = blueprint_matrix(data)
        countries = adj_matrix.columns

        data_lang = data_lang[data_lang['iso3_i'].isin(countries) & data_lang['iso3_j'].isin(countries)]
        for _, row in data_lang.iterrows():
            adj_matrix.loc[row['iso3_i'], row['iso3_j']] = row['col']

        adj_matrix = adj_matrix.astype('int')
        output_path = path_to_save / "lang_matrix.csv"
        adj_matrix.to_csv(output_path, index=True)
        print(f"Lang matrix successfully created: {output_path}")
    except Exception as e:
        print(f"Error creating language file: {e}")
        return None


def border_matrix(data, path_to_save):
    try:
        path_to_save = Path(path_to_save)
        borders = pd.read_csv(Path("initial_data_files") / "borders.csv")
        borders.replace({"country_name": {"Namibia": "NA"}, "country_border_name": {"Namibia": "NA"}}, inplace=True)
        borders.dropna(inplace=True)

        ISO2_ISO3 = pd.read_csv(Path("initial_data_files") / "ISO2_ISO3.csv", delimiter=';')
        iso2_iso3_dict = dict(zip(ISO2_ISO3['ISO2'], ISO2_ISO3['ISO3']))
        iso2_iso3_dict['NA'] = 'NAM'

        borders['country_code'] = borders['country_code'].map(iso2_iso3_dict)
        borders['country_border_code'] = borders['country_border_code'].map(iso2_iso3_dict)

        data.year = data.year.astype(int)
        countries = sorted(set(data['iso3codefrom']).union(data['iso3codeto']))
        adj_matrix = pd.DataFrame(0, columns=countries, index=countries, dtype=int)

        borders = borders[borders['country_code'].isin(countries) & borders['country_border_code'].isin(countries)]
        adj_matrix.loc[borders['country_code'], borders['country_border_code']] = 1

        output_path = path_to_save / "border_matrix.csv"
        adj_matrix.to_csv(output_path, index=True)
        print(f"Border matrix successfully created: {output_path}")
    except Exception as e:
        print(f"Error creating border file: {e}")
        return None


def create_structure_df(data):
    try:
        years = list(range(1998, 2019))
        data.year = data.year.astype(int)
        countries = sorted(set(data['iso3codefrom']).union(data['iso3codeto']))
        df = pd.DataFrame(index=countries, columns=years)
        return df
    except Exception as e:
        print(f"Error creating structure file: {e}")
        return None


def academ_freedom():
    try:
        df = pd.read_csv(Path("initial_data_files") / "V-Dem-CY-Full+Others-v14.csv", usecols=['country_text_id', 'year', 'v2xca_academ'], low_memory=False)
        df = df[df['year'] > 1997]
        academ_freedom_dict = {(row['year'], row['country_text_id']): row['v2xca_academ'] for _, row in df.iterrows()}
        print(f"Successfully processed academ_freedom data dictionary")
        return academ_freedom_dict
    except Exception as e:
        print(f"Error creating academ freedom file: {e}")
        return None


def uni_ratings():
    try:
        shrank = pd.read_csv(Path("initial_data_files") / "shanghai-world-university-ranking.csv", delimiter=';')
        ranksh_df = shrank.groupby(['Year', 'ISO3 CODE'])['Country'].count().reset_index()
        dict = {(row['Year'], row['ISO3 CODE']): row['Country'] for _, row in ranksh_df.iterrows()}
        return dict
        print('Successfully processed university ranking data into dictionary')
    except Exception as e:
        print(f"Error creating uni ratings dictionary: {e}")
        return None


def country_iso_dictionary(data):
    try:
        country_isofrom = zip(data['countrynamefrom'], data['iso3codefrom'])
        country_isofrom = set(country_isofrom)

        country_isoto = zip(data['countrynameto'], data['iso3codeto'])
        country_isoto = set(country_isoto)

        country_iso = country_isofrom.union(country_isoto)
        country_iso = set(country_iso)
        dictionary = dict(country_iso)
        print('Successfully created country iso dictionary')
        return dictionary
    except Exception as e:
        print(f"Error creating country iso dictionary: {e}")
        return None


def conflicts(country_iso):
    try:
        locations_conf_df = pd.read_excel(Path("initial_data_files") / "loc_conflicts.xlsx")
        locations_conf_df['iso3'] = locations_conf_df['location'].map(country_iso)

        locations_conf_dict = {}
        for year in range(1998, 2018):
            yearly_df = locations_conf_df[locations_conf_df['year'] == year].groupby('iso3').count()
            locations_conf_dict.update({(year, index): row['location'] for index, row in yearly_df.iterrows()})
        print('Successfully created conflicts dictionary')
        return locations_conf_dict
    except Exception as e:
        print(f"Error creating conflicts file: {e}")
        return None


def create_nodes_files(data, academ_freedom_dict, rank_dict, locations_conf_dict, path_to_save):
    data.year = data.year.astype(int)
    countries = sorted(list(set(list(data['iso3codefrom'])+list(data['iso3codeto']))))
    countries_df = pd.DataFrame(countries, columns=['iso3code'])

    for year in range(data['year'].min(), data['year'].max() + 1):
        try:
            data_year = data[data['year'] == year].copy()

            from_cols = ['iso3codefrom', 'regionfrom', 'gdp_per_capitafrom', 'populationfrom', 'paddedpopfrom',
                          'incomelevelfrom']
            to_cols = ['iso3codeto', 'regionto', 'gdp_per_capitato', 'populationto', 'paddedpopto',
                      'incomelevelto']
            rename_cols = {
                'iso3codefrom': 'iso3code', 'regionfrom': 'region', 'gdp_per_capitafrom': 'gdp_per_capita',
                'populationfrom': 'population', 'paddedpopfrom': 'paddedpop',
                'incomelevelfrom': 'incomelevel'
            }
            from_data = data_year[from_cols].rename(columns=rename_cols).drop_duplicates()
            to_data = data_year[to_cols].rename(columns={col: rename_cols[col.replace('to', 'from')] for col in to_cols}).drop_duplicates()

            nodes = pd.concat([from_data, to_data], axis=0).drop_duplicates()
            nodes = nodes.merge(countries_df, how='right', on='iso3code').rename(columns={'iso3code': 'label'})
            nodes['id'] = nodes['label']
            nodes['year'] = year
            nodes.fillna(0, inplace=True)

            nodes['freedom'] = nodes['label'].map(
                {country: value for (yearr, country), value in academ_freedom_dict.items() if yearr == year})
            nodes['top500'] = nodes['label'].map(
                {country: value for (yearr, country), value in rank_dict.items() if yearr == year})
            nodes['conflict'] = nodes['label'].map(
                {country: value for (yearr, country), value in locations_conf_dict.items() if yearr == year})

            if year > 1998:
                nodes['conflict_lag1'] = nodes['label'].map(
                    {country: value for (yearr, country), value in locations_conf_dict.items() if yearr == year - 1})
            else:
                nodes['conflict_lag1'] = 1

            nodes.fillna(0, inplace=True)
            output_file = path_to_save / f'nodes_{year}.csv'
            nodes.to_csv(output_file, index=False)
            print(f"Nodes file {year} successfully created: {output_file}")
        except Exception as e:
            print(f"Error creating nodes file {year}: {e}")
            return None


def handling_missing_values_nodes(data, val, path_to_save):
    try:
        path_to_save = Path(path_to_save)

        from_df = data[['iso3codefrom', 'year', f'{val}from']].rename(
            columns={'iso3codefrom': 'iso3code', f'{val}from': val})
        to_df = data[['iso3codeto', 'year', f'{val}to']].rename(columns={'iso3codeto': 'iso3code', f'{val}to': val})

        concated = pd.concat([from_df, to_df]).drop_duplicates()
        struct = create_structure_df(data)
        struct.update(concated.pivot(index='iso3code', columns='year', values=val))

        struct = struct.apply(pd.to_numeric, errors='coerce').ffill(axis=1).bfill(axis=1)
        struct = struct.apply(lambda col: col.fillna(col.median()), axis=0)


        for year in range(data['year'].min(), data['year'].max() + 1):
            nodes_file = path_to_save / f'nodes_{year}.csv'
            nodes = pd.read_csv(nodes_file)
            nodes['year'] = nodes['year'].astype(int)

            nodes[val] = nodes.apply(
                lambda row: struct.loc[row['label'], row['year']]
                if row['label'] in struct.index and row['year'] in struct.columns else np.nan, axis=1
            )
            if val == 'gdp_per_capita':
                nodes['log_gdp_per_capita'] = np.log10(nodes['gdp_per_capita'].replace(0, np.nan))
            nodes.to_csv(nodes_file, index=False)
            print(f"Nodes file {year} missed calues are filles: {nodes_file}")
    except Exception as e:
        print(f"Error filling missing values in nodes file {val}: {e}")
        return None


def main():
    try:
        path_to_save = Path("processed_data")
        path_to_save.mkdir(exist_ok=True)

        print("Loading data...")
        try:
            data = pd.read_csv(Path("initial_data_files") / "scopus_2024_migration.csv")
            print("Data loaded!")
        except Exception as e:
            print(f"Error loading data: {e}")


        print("Creating geo file...")
        geo_countries = create_geo_file(path_to_save)

        print("Creating distance matrix...")
        distance_matrix(geo_countries, data, path_to_save)

        print("Processing academ_freedom data...")
        academ_freedom_dict = academ_freedom()

        print("Processing university rankings...")
        rank_dict = uni_ratings()

        print("Processing country-iso dictionary...")
        country_iso = country_iso_dictionary(data)

        print("Processing conflicts data...")
        locations_conf_dict = conflicts(country_iso)

        print("Creating language matrix...")
        language_matrix(data, path_to_save)

        print("Creating border matrix...")
        border_matrix(data, path_to_save)

        print("Creating nodes files...")
        create_nodes_files(data, academ_freedom_dict, rank_dict, locations_conf_dict, path_to_save)

        print("Handling missing values in nodes files...")
        for val in ['gdp_per_capita', 'paddedpop', 'population']:
            handling_missing_values_nodes(data, val, path_to_save)

        print("Processing complete!")
    except Exception as e:
        print(f"Error in main execution: {e}")


if __name__ == "__main__":
    main()