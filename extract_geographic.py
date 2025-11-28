import pandas as pd
import os

def extract_geographic_data():
    """Extract geographic data from CFSVA survey files for mapping"""

    # File paths
    hh_file = "Microdata1111/Microdata/csvfile/CFSVA2024_HH data.csv"
    child_file = "Microdata1111/Microdata/csvfile/CFSVA2024_HH_CHILD_6_59_MONTHS.csv"
    women_file = "Microdata1111/Microdata/csvfile/CFSVA2024_HH_WOMEN_15_49_YEARS.csv"

    geographic_data = {}

    try:
        # Load household data
        print("Loading household data...")
        hh_data = pd.read_csv(hh_file)

        # Extract geographic columns from household data
        geo_cols_hh = {}
        if 'S0_C_Prov' in hh_data.columns:
            geo_cols_hh['province'] = hh_data['S0_C_Prov']
        if 'S0_D_Dist' in hh_data.columns:
            geo_cols_hh['district'] = hh_data['S0_D_Dist']
        if 'UrbanRural' in hh_data.columns:
            geo_cols_hh['urban_rural'] = hh_data['UrbanRural']

        geographic_data['household'] = pd.DataFrame(geo_cols_hh)

        # Load child data
        print("Loading child data...")
        child_data = pd.read_csv(child_file)

        # Extract geographic columns from child data
        geo_cols_child = {}
        if 'S0_C_Prov' in child_data.columns:
            geo_cols_child['province'] = child_data['S0_C_Prov']
        if 'S0_D_Dist' in child_data.columns:
            geo_cols_child['district'] = child_data['S0_D_Dist']
        if 'UrbanRural' in child_data.columns:
            geo_cols_child['urban_rural'] = child_data['UrbanRural']

        geographic_data['child'] = pd.DataFrame(geo_cols_child)

        # Load women data
        print("Loading women data...")
        women_data = pd.read_csv(women_file)

        # Extract geographic columns from women data
        geo_cols_women = {}
        if 'S0_C_Prov' in women_data.columns:
            geo_cols_women['province'] = women_data['S0_C_Prov']
        if 'S0_D_Dist' in women_data.columns:
            geo_cols_women['district'] = women_data['S0_D_Dist']
        if 'UrbanRural' in women_data.columns:
            geo_cols_women['urban_rural'] = women_data['UrbanRural']

        geographic_data['women'] = pd.DataFrame(geo_cols_women)

        # Create summary statistics
        print("Creating geographic summaries...")

        # Province distribution
        province_summary = {}
        for dataset_name, data in geographic_data.items():
            if 'province' in data.columns:
                province_counts = data['province'].value_counts()
                province_summary[dataset_name] = province_counts

        # District distribution
        district_summary = {}
        for dataset_name, data in geographic_data.items():
            if 'district' in data.columns:
                district_counts = data['district'].value_counts()
                district_summary[dataset_name] = district_counts

        # Urban/Rural distribution
        urban_rural_summary = {}
        for dataset_name, data in geographic_data.items():
            if 'urban_rural' in data.columns:
                urban_rural_counts = data['urban_rural'].value_counts()
                urban_rural_summary[dataset_name] = urban_rural_counts

        # Save geographic data to CSV files
        print("Saving geographic data...")

        for dataset_name, data in geographic_data.items():
            output_file = f"geographic_{dataset_name}.csv"
            data.to_csv(output_file, index=False)
            print(f"Saved {output_file} with {len(data)} records")

        # Save summary statistics
        with pd.ExcelWriter('geographic_summary.xlsx') as writer:
            # Province summaries
            for dataset_name, summary in province_summary.items():
                summary.to_frame(f'{dataset_name}_province').to_excel(writer, sheet_name=f'{dataset_name}_province')

            # District summaries
            for dataset_name, summary in district_summary.items():
                summary.to_frame(f'{dataset_name}_district').to_excel(writer, sheet_name=f'{dataset_name}_district')

            # Urban/Rural summaries
            for dataset_name, summary in urban_rural_summary.items():
                summary.to_frame(f'{dataset_name}_urban_rural').to_excel(writer, sheet_name=f'{dataset_name}_urban_rural')

        print("Geographic data extraction completed!")
        print("\nSummary:")
        print(f"- Household records: {len(geographic_data['household'])}")
        print(f"- Child records: {len(geographic_data['child'])}")
        print(f"- Women records: {len(geographic_data['women'])}")

        # Display unique values for mapping
        print("\nUnique Provinces:")
        if 'province' in geographic_data['household'].columns:
            print(geographic_data['household']['province'].unique())

        print("\nUnique Districts:")
        if 'district' in geographic_data['household'].columns:
            print(geographic_data['household']['district'].unique()[:20])  # Show first 20
            print(f"... and {len(geographic_data['household']['district'].unique()) - 20} more districts")

        return geographic_data

    except Exception as e:
        print(f"Error extracting geographic data: {e}")
        return None

if __name__ == "__main__":
    extract_geographic_data()