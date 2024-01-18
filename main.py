from data_profile import read_csv_data, DataPreparation, DataProfile
from train_eval import TrainModel

def main()->None:

    csv_file_path = "cfhtlens.csv"
    data = read_csv_data(csv_file_path)
    prep_data, prep_data_profile = DataPreparation(data).process_data()

    profileInstance = DataProfile(prep_data_profile)
    profileInstance.generate_histogram()
    profileInstance.generate_pdf()
    
    output_csv_path = 'prep_data.csv'
    prep_data.to_csv(output_csv_path, index=False)

    eval_instance = TrainModel(prep_data)
    eval_instance.load_data()
    eval_instance.train_model()
    eval_instance.evaluate_model()

    print("Accuracy of entire test set",eval_instance.accuracy_dict_entire_test_set)
    print("Accuracy of missing test set",eval_instance.accuracy_dict_missing_values)


if __name__ == "__main__":
    main()
