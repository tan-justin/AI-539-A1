import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer

"""
Type: Class
Name: ExtraCredit
Purpose: The ExtraCredit class which contains functions for preprocessing the dataset, splitting it and using methods for handling
         missing data
Parameters: Pandas Dataframe, float Threshold, int random_seed
---------------------------------------------------------------------------------------------------------------------------------
Type: Function
Name: load_data_extra
Purpose: Prepare the training and testing sets as well as their truth labels. The training set will also contain items with missing
         values
Parameters: None
---------------------------------------------------------------------------------------------------------------------------------
Type: Function
Name: method_C
Purpose: Perform feature omission on both the training set and testing set, based on the features that were confirmed to have missing
         values in the original dataset
Parameters: None
---------------------------------------------------------------------------------------------------------------------------------
Type: Function
Name: method_D
Purpose: Perform mean imputation on the items with missing values. The mean imputed is strictly dependent on the feature the missing
         value falls under and is obtained from the non-missing items in the training set. Imputation is performed on both training
         and testing sets
Parameters: None
---------------------------------------------------------------------------------------------------------------------------------
Type: Function
Name: method_E
Purpose: Perform median imputation on the items with missing values. The median imputed is strictly dependent on the feature the missing
         value falls under and is obtained from the non-missing items in the training set. Imputation is performed on both training
         and testing sets
Parameters: None
---------------------------------------------------------------------------------------------------------------------------------
Type: Function
Name: method_F
Purpose: Perform KNN imputation on the items with missing values. The imputed value is strictly dependent on the KNN of the missing
         item and the respective feature. Imputation is performed on both training and testing sets.
Parameters: None
---------------------------------------------------------------------------------------------------------------------------------
Type: Function
Name: train_and_pred
Purpose: Call the above method_* functions
Parameters: None
"""
class ExtraCredit:

    def __init__(self, data, threshold = 0.5, random_seed = 0):
        self.data = data
        self.threshold = threshold
        self.random_seed = random_seed
        self.model = RandomForestClassifier(random_state = random_seed)
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.accuracy_dict_entire_test_set = {}
        self.feature_labels = None
        self.columns_missing = None

    def load_data_extra(self):

        data = self.data.copy()
        feature_labels = data.columns.tolist()
        feature_labels.pop(0)
        self.feature_labels = feature_labels
        Xy = data.to_numpy()
        X = Xy[:,1:]
        y = (Xy[:,0] >= self.threshold).astype(int) #sets to 0 if star and 1 if galaxy
        self.x_train, self.x_test, self.y_train, self.y_test = \
            train_test_split(X, y, train_size = 3000, random_state = self.random_seed)
        missing_values = data.isnull().any()
        columns_missing = data.columns[missing_values].tolist()
        self.columns_missing = columns_missing

    def method_C(self):

        x_train = self.x_train.copy()
        y_train = self.y_train.copy()
        x_test = self.x_test.copy()
        y_test = self.y_test.copy()
        labels_C = self.feature_labels.copy()
        rebuilt_x_train = pd.DataFrame(x_train, columns = labels_C)
        rebuilt_x_test = pd.DataFrame(x_test, columns = labels_C)
        rebuilt_x_train = rebuilt_x_train.drop(columns = self.columns_missing, axis = 1)
        rebuilt_x_test = rebuilt_x_test.drop(columns = self.columns_missing, axis = 1)
        model = RandomForestClassifier(random_state = self.random_seed)
        model.fit(rebuilt_x_train, y_train)
        y_pred = model.predict(rebuilt_x_test)
        accuracy = accuracy_score(y_test, y_pred)
        self.accuracy_dict_entire_test_set['C'] = accuracy
    
    def method_D(self):
    
        x_train = self.x_train.copy()
        x_test = self.x_test.copy()
        y_train = self.y_train.copy()
        y_test = self.y_test.copy()
        feature_labels = self.feature_labels.copy()
        rebuilt_x_missing_train = pd.DataFrame(x_train, columns = feature_labels)
        rebuilt_x_missing_test = pd.DataFrame(x_test, columns = feature_labels)
        mean_feature_dict = {}
        for column in self.columns_missing:
            mean_feature_dict[column] = rebuilt_x_missing_train[column].mean(skipna = True)
        for column in mean_feature_dict:
            rebuilt_x_missing_train[column] = rebuilt_x_missing_train[column].fillna(mean_feature_dict[column])
            rebuilt_x_missing_test[column] = rebuilt_x_missing_test[column].fillna(mean_feature_dict[column])
        model = RandomForestClassifier(random_state = self.random_seed)
        model.fit(rebuilt_x_missing_train, y_train)
        y_pred = model.predict(rebuilt_x_missing_test)
        accuracy = accuracy_score(y_test, y_pred)
        self.accuracy_dict_entire_test_set['D'] = accuracy

    def method_E(self):
    
        x_train = self.x_train.copy()
        x_test = self.x_test.copy()
        y_train = self.y_train.copy()
        y_test = self.y_test.copy()
        feature_labels = self.feature_labels.copy()
        rebuilt_x_missing_train = pd.DataFrame(x_train, columns = feature_labels)
        rebuilt_x_missing_test = pd.DataFrame(x_test, columns = feature_labels)
        median_feature_dict = {}
        for column in self.columns_missing:
            median_feature_dict[column] = rebuilt_x_missing_train[column].median(skipna = True)
        for column in median_feature_dict:
            rebuilt_x_missing_train[column] = rebuilt_x_missing_train[column].fillna(median_feature_dict[column])
            rebuilt_x_missing_test[column] = rebuilt_x_missing_test[column].fillna(median_feature_dict[column])
        model = RandomForestClassifier(random_state = self.random_seed)
        model.fit(rebuilt_x_missing_train, y_train)
        y_pred = model.predict(rebuilt_x_missing_test)
        accuracy = accuracy_score(y_test, y_pred)
        self.accuracy_dict_entire_test_set['E'] = accuracy

    def method_F(self):

        x_train = self.x_train.copy()
        x_test = self.x_test.copy()
        y_train = self.y_train.copy()
        y_test = self.y_test.copy()
        feature_labels = self.feature_labels.copy()
        rebuilt_x_missing_train = pd.DataFrame(x_train, columns = feature_labels)
        rebuilt_x_missing_test = pd.DataFrame(x_test, columns = feature_labels)
        knn_imputer = KNNImputer(n_neighbors = 30)
        knn_imputer.fit(rebuilt_x_missing_train)
        rebuilt_x_train_imputed = knn_imputer.transform(rebuilt_x_missing_train)
        rebuilt_x_test_imputed =  knn_imputer.transform(rebuilt_x_missing_test)
        model = RandomForestClassifier(random_state = self.random_seed)
        model.fit(rebuilt_x_train_imputed, y_train)
        y_pred = model.predict(rebuilt_x_test_imputed)
        accuracy = accuracy_score(y_test, y_pred)
        self.accuracy_dict_entire_test_set['F'] = accuracy

    def train_and_pred(self):

        self.method_C()
        self.method_D()
        self.method_E()
        self.method_F()

        print("Accuracy: ",self.accuracy_dict_entire_test_set)

    


