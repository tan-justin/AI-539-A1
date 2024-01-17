from data_profile import read_csv_data, DataPreparation, DataProfile

def main()->None:

    csv_file_path = "cfhtlens.csv"
    data = read_csv_data(csv_file_path)
    prep_data, prep_data_profile = DataPreparation(data).process_data()

    profileInstance = DataProfile(prep_data_profile)
    profileInstance.generate_histogram()
    profileInstance.generate_pdf()
    
    output_csv_path = 'prep_data.csv'
    prep_data.to_csv(output_csv_path, index=False)

if __name__ == "__main__":
    main()
