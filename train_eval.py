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

    def load_data(self):

        data = self.data
        Xy = data.to_numpy()
        X = Xy[:,1:]
        y = (Xy[:,0] >= self.threshold).astype(int) #sets to 0 if star and 1 if galaxy
        missing = np.sum(np.isnan(X), axis = 1) > 0
        self.x_train, self.x_test, self.y_train, self.y_test = \
            train_test_split(X[~missing], y[~missing], train_size = 3000, random_state = self.random_state)
        self.x_missing = X[missing]
        self.y_missing = y[missing]
        self.missing = missing

    def train_model(self):

        self.model.fit(self.x_train, self.y_train)

#x_test = np.concatenate((self.x_test, X[missing]))
#y_test = np.concatenate((self.y_test, y[missing]))

    def evaluate_model(self):
        
        method = ['A','B','C','D','E']

        for method in method:

            x_test = self.x_test.copy()
            y_test = self.y_test.copy()
            x_missing = self.x_missing.copy()
            y_missing = self.y_missing.copy()
            
            if method == 'A': #method: Abstention

                x_test_full = np.concatenate(x_test, x_missing)
                y_test_full = np.concatenate(y_test, y_missing) #truth labels of both y
                y_pred = self.model.predict(x_test_full)
                missing_indices = np.where(self.missing)[0]
                y_pred[missing_indices] = -1 
                #we know that it will never be classified as -1, hence we can trigger a wrong prediction for items with missing values

                accuracy = accuracy_score(y_test_full, y_pred)
                
                self.accuracy_dict_entire_test_set[method] = accuracy

                #because items with missing values are treated as errors, we can ignore it and give it a 0 accuracy
                self.accuracy_dict_missing_values[method] = 0.0 
            
            if method == 'B':

                majority_class = np.bincount(self.y_train).argmax() #gets the majority class here
                y_pred_majority = np.full_like(y_missing, fill_value = majority_class) 
                #create a new np array that is of dimension y_missing but filled with the majority class label

                #combine both x_test with x_missing and combine y_test with y_missing here

                x_test_full = np.concatenate(x_test, x_missing) 
                y_test_full = np.concatenate(y_test, y_missing)

                y_pred = self.model.predict(x_test)
                y_pred[self.missing] = y_pred_majority

                accuracy = accuracy_score(y_test_full, y_pred)
                self.accuracy_dict_entire_test_set[method] = accuracy

                #missing only now
                #we compare y_pred_majority predictions with y_missing
                accuracy_missing = accuracy_score(y_missing, y_pred_majority)
                self.accuracy_dict_missing_values[method] = accuracy_missing



                



