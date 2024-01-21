from data_profile import read_csv_data, DataPreparation, DataProfile
from train_eval import TrainModel
from extra_credit import ExtraCredit 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def main()->None:

    csv_file_path = "cfhtlens.csv"
    data = read_csv_data(csv_file_path)
    prep_data, prep_data_profile = DataPreparation(data).process_data() #processing data for profiling purposes

    profileInstance = DataProfile(prep_data_profile)
    profileInstance.generate_histogram()
    profileInstance.generate_pdf()
    
    output_csv_path = 'prep_data.csv' #generating the new csv for analysis purposes (MCAR/MAR/MMAR) 
    prep_data.to_csv(output_csv_path, index = False) 

    eval_instance = TrainModel(prep_data) 
    eval_instance.load_data()
    trained_model = eval_instance.train_model()
    eval_instance.evaluate_model()

    print("Accuracy of entire test set: ",eval_instance.accuracy_dict_entire_test_set)
    print("Accuracy of missing test set: ",eval_instance.accuracy_dict_missing_values)

    birthday_file_path = "BirthdayStar.csv"
    data_bday = read_csv_data(birthday_file_path)
    prep_data_bday, prep_data_bday_profile = DataPreparation(data_bday).process_data()
    prep_data_bday_Xy = prep_data_bday.to_numpy()
    prep_data_bday_X = prep_data_bday_Xy[:,1:]
    prep_data_bday_y = (prep_data_bday_Xy[:,0] >= 0.5).astype(int) #creating the binary classification for the truth label
    y_pred = trained_model.predict(prep_data_bday_X)

    if accuracy_score(prep_data_bday_y, y_pred) == 1.0:
        print("Classified birthday sky object correctly: True")
    else:
        print("Classified birthday sky object correctly: False")

    print("Extra credit: ")
    extra_credit_instance = ExtraCredit(prep_data)
    extra_credit_instance.load_data_extra()
    extra_credit_instance.train_and_pred()

if __name__ == "__main__":
    main()
