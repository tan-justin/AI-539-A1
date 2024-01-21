import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


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

    def load_data_extra(self):

        data = self.data.copy()
        feature_labels = data.columns.tolist()
        self.feature_labels = feature_labels
        Xy = data.to_numpy()
        X = Xy[:,1:]
        y = (Xy[:,0] >= self.threshold).astype(int) #sets to 0 if star and 1 if galaxy
        self.x_train, self.x_test, self.y_train, self.y_test = \
            train_test_split(X, y, train_size = 3000, random_state = self.random_seed)
        
    def method_C(self):

        x_train = self.x_train.copy()
        y_train = self.y_train.copy()
        x_test = self.x_test.copy()
        y_test = self.y_test.copy()
        labels_C = self.feature_labels.copy()
        labels_C.pop(0)
        rebuilt_x_train = pd.DataFrame(x_train, columns = labels_C)
        rebuilt_x_test = pd.DataFrame(x_test, columns = labels_C)
        missing_values_train = rebuilt_x_train.isnull().any()
        missing_values_test = rebuilt_x_test.isnull().any()
        columns_to_drop_train = missing_values_train.index[missing_values_train].tolist()
        columns_to_drop_test = missing_values_test.index[missing_values_test].tolist()
        #this if else statement is in the event the train test split results in one set containing all of the items w/ missing values
        if len(columns_to_drop_train) == len(columns_to_drop_test): 
            columns_to_drop = columns_to_drop_train
        elif len(columns_to_drop_train) < len(columns_to_drop_test):
            columns_to_drop = columns_to_drop_test
        else:
            columns_to_drop = columns_to_drop_train
        rebuilt_x_train = rebuilt_x_train.drop(columns = columns_to_drop, axis = 1)
        rebuilt_x_test = rebuilt_x_test.drop(columns = columns_to_drop, axis = 1)
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
        feature_labels.pop(0)
        rebuilt_x_missing_train = pd.DataFrame(x_train, columns = feature_labels)
        rebuilt_x_missing_test = pd.DataFrame(x_test, columns = feature_labels)
        mean_feature_dict = {}
        missing_values_train = rebuilt_x_missing_train.isnull().any()
        missing_values_test = rebuilt_x_missing_test.isnull().any()
        columns_missing_train = rebuilt_x_missing_train.columns[missing_values_train].tolist()
        columns_missing_test = rebuilt_x_missing_test.columns[missing_values_test].tolist()
        #this if else statement is in the event the train test split results in one set containing all of the items w/ missing values
        if len(columns_missing_train) == len(columns_missing_test): 
            columns_missing = columns_missing_train
        elif len(columns_missing_train) < len(columns_missing_test):
            columns_missing = columns_missing_test
        else:
            columns_missing = columns_missing_train

        for column in columns_missing:
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
        feature_labels.pop(0)
        rebuilt_x_missing_train = pd.DataFrame(x_train, columns = feature_labels)
        rebuilt_x_missing_test = pd.DataFrame(x_test, columns = feature_labels)

        median_feature_dict = {}
        missing_values_train = rebuilt_x_missing_train.isnull().any()
        missing_values_test = rebuilt_x_missing_test.isnull().any()
        columns_missing_train = rebuilt_x_missing_train.columns[missing_values_train].tolist()
        columns_missing_test = rebuilt_x_missing_test.columns[missing_values_test].tolist()
        #this if else statement is in the event the train test split results in one set containing all of the items w/ missing values
        if len(columns_missing_train) == len(columns_missing_test): 
            columns_missing = columns_missing_train
        elif len(columns_missing_train) < len(columns_missing_test):
            columns_missing = columns_missing_test
        else:
            columns_missing = columns_missing_train
        for column in columns_missing:
            median_feature_dict[column] = rebuilt_x_missing_train[column].median(skipna = True)
        for column in median_feature_dict:
            rebuilt_x_missing_train[column] = rebuilt_x_missing_train[column].fillna(median_feature_dict[column])
            rebuilt_x_missing_test[column] = rebuilt_x_missing_test[column].fillna(median_feature_dict[column])
        model = RandomForestClassifier(random_state = self.random_seed)
        model.fit(rebuilt_x_missing_train, y_train)
        y_pred = model.predict(rebuilt_x_missing_test)
        accuracy = accuracy_score(y_test, y_pred)
        self.accuracy_dict_entire_test_set['E'] = accuracy

    def train_and_pred(self):

        self.method_C()
        self.method_D()
        self.method_E()

        print("Accuracy: ",self.accuracy_dict_entire_test_set)

    


