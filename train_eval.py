import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score

class TrainModel:

    def __init__(self, data, threshold = 0.5, random_seed = 0):

        self.data = data
        self.threshold = threshold
        self.random_seed = random_seed
        self.model = RandomForestClassifier(random_state = random_seed)
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.x_missing = None
        self.y_missing = None
        self.missing = None
        self.accuracy_dict_entire_test_set = {}
        self.accuracy_dict_missing_values = {}
        self.feature_labels = None

    def load_data(self):

        data = self.data.copy()
        feature_labels = data.columns.tolist()
        Xy = data.to_numpy()
        X = Xy[:,1:]
        y = (Xy[:,0] >= self.threshold).astype(int) #sets to 0 if star and 1 if galaxy
        missing = np.sum(np.isnan(X), axis = 1) > 0
        self.x_train, self.x_test, self.y_train, self.y_test = \
            train_test_split(X[~missing], y[~missing], train_size = 3000, random_state = self.random_seed)
        self.x_missing = X[missing]
        self.y_missing = y[missing]
        self.missing = missing
        self.feature_labels = feature_labels

    def train_model(self):

        self.model.fit(self.x_train, self.y_train)

    def evaluate_model(self):
        
        method = ['A','B','C','D','E']

        for method in method:

            x_test = self.x_test.copy()
            y_test = self.y_test.copy()
            x_missing = self.x_missing.copy()
            y_missing = self.y_missing.copy()
            
            if method == 'A': #method: Abstention

                x_train = self.x_train.copy()
                imputer = SimpleImputer(strategy='mean')
                x_train_imputed = imputer.fit_transform(x_train)
                x_test_full = np.concatenate((x_test, x_missing))
                y_test_full = np.concatenate((y_test, y_missing)) #truth labels of both y
                x_test_full_imputed = imputer.transform(x_test_full)

                y_pred = self.model.predict(x_test_full_imputed)
                num_missing_items = x_missing.shape[0]
                y_pred[-num_missing_items:] = -1 
                #we know that it will never be classified as -1, hence we can trigger a wrong prediction for items with missing values

                accuracy = accuracy_score(y_test_full, y_pred)
                self.accuracy_dict_entire_test_set[method] = accuracy

                #because items with missing values are treated as errors, we can ignore it and give it a 0 accuracy
                missing_items_array = np.full((num_missing_items, ), -1)
                accuracy_missing = accuracy_score(y_missing, missing_items_array)
                self.accuracy_dict_missing_values[method] = accuracy_missing

                print("Method A succeeded")
            
            if method == 'B': #majority inference

                majority_class = np.bincount(self.y_train).argmax() #gets the majority class here
                y_pred_majority = np.full_like(y_missing, fill_value = majority_class) 
                #create a new np array that is of dimension y_missing but filled with the majority class label

                #combine both x_test with x_missing and combine y_test with y_missing here

                x_test_full = np.concatenate((x_test, x_missing)) 
                y_test_full = np.concatenate((y_test, y_missing))

                y_pred_non_missing = self.model.predict(x_test)
                y_pred = np.concatenate((y_pred_non_missing, y_pred_majority))

                accuracy = accuracy_score(y_test_full, y_pred)
                self.accuracy_dict_entire_test_set[method] = accuracy

                #missing only now
                #we compare y_pred_majority predictions with y_missing
                accuracy_missing = accuracy_score(y_missing, y_pred_majority)
                self.accuracy_dict_missing_values[method] = accuracy_missing

                print("Method B succeeded")
            
            if method == 'C':
            # omit any features with missing values (based on report generated, the features to be eliminated should be
            # )
                x_test_full = np.concatenate((x_test, x_missing))
                y_test_full = np.concatenate((y_test, y_missing))
                labels_C = self.feature_labels.copy()
                labels_C.pop(0) #remove the truth label column name from the list
                rebuilt_x_test = pd.DataFrame(x_test_full, columns = labels_C)
                missing_values = rebuilt_x_test.isnull().any()
                columns_to_drop = missing_values.index[missing_values].tolist()
                rebuilt_x_test = rebuilt_x_test.drop(columns = columns_to_drop, axis = 1) #omitting features with missing values here from the testing set

                x_train = self.x_train.copy()
                y_train = self.y_train.copy() 
                #shouldn't require a copy for y_train, but for the sake of the next two methods, 
                #we'll use copy to prevent modifications to the base truth label set
                rebuilt_x_train = pd.DataFrame(x_train, columns = labels_C)
                rebuilt_x_train = rebuilt_x_train.drop(columns = columns_to_drop, axis = 1) #omitting from training set
                
                model = RandomForestClassifier(random_state = self.random_seed)
                model.fit(rebuilt_x_train, y_train)

                y_pred = model.predict(rebuilt_x_test)
                accuracy = accuracy_score(y_test_full, y_pred)
                self.accuracy_dict_entire_test_set[method] = accuracy

                rebuilt_x_missing = pd.DataFrame(x_missing, columns = labels_C)
                rebuilt_x_missing = rebuilt_x_missing.drop(columns = columns_to_drop, axis = 1)
                y_pred_missing = model.predict(rebuilt_x_missing)
                accuracy_missing = accuracy_score(y_missing, y_pred_missing)
                self.accuracy_dict_missing_values[method] = accuracy_missing

                print("Method C suceeded")

            if method == 'D':

                feature_labels = self.feature_labels.copy()
                feature_labels.pop(0)
                rebuilt_x_missing = pd.DataFrame(x_missing, columns = feature_labels)
                missing_values = rebuilt_x_missing.isnull().any()
                columns_missing_values = missing_values.index[missing_values].tolist()
                
                x_train = self.x_train.copy()
                rebuilt_x_train = pd.DataFrame(x_train, columns = feature_labels)
                mean_feature_dict = {}

                for column in columns_missing_values:

                    mean_feature_dict[column] = rebuilt_x_train[column].mean()

                
                for column in mean_feature_dict:

                    rebuilt_x_missing[column] = rebuilt_x_missing[column].fillna(mean_feature_dict[column])

                x_test_full = np.concatenate((x_test, rebuilt_x_missing))
                y_test_full = np.concatenate((y_test, y_missing))

                y_pred = self.model.predict(x_test_full)
                accuracy = accuracy_score(y_test_full, y_pred)
                self.accuracy_dict_entire_test_set[method] = accuracy

                y_pred_missing = self.model.predict(rebuilt_x_missing.values)
                accuracy_missing = accuracy_score(y_missing, y_pred_missing)
                self.accuracy_dict_missing_values[method] = accuracy_missing

                print("Method D succeeded")
            
            if method == 'E':

                feature_labels = self.feature_labels.copy()
                feature_labels.pop(0)
                rebuilt_x_missing = pd.DataFrame(x_missing, columns = feature_labels)
                missing_values = rebuilt_x_missing.isnull().any()
                columns_missing_values = missing_values.index[missing_values].tolist()
                
                x_train = self.x_train.copy()
                rebuilt_x_train = pd.DataFrame(x_train, columns = feature_labels)
                median_feature_dict = {}

                for column in columns_missing_values:

                    median_feature_dict[column] = rebuilt_x_train[column].median()

                
                for column in median_feature_dict:

                    rebuilt_x_missing[column] = rebuilt_x_missing[column].fillna(median_feature_dict[column])

                x_test_full = np.concatenate((x_test, rebuilt_x_missing))
                y_test_full = np.concatenate((y_test, y_missing))

                y_pred = self.model.predict(x_test_full)
                accuracy = accuracy_score(y_test_full, y_pred)
                self.accuracy_dict_entire_test_set[method] = accuracy

                y_pred_missing = self.model.predict(rebuilt_x_missing.values)
                accuracy_missing = accuracy_score(y_missing, y_pred_missing)
                self.accuracy_dict_missing_values[method] = accuracy_missing

                print("Method E succeeded")
            

            else:
                continue
        
        




                    
























                



