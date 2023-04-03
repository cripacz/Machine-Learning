import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

class KNNRegressor_weighted():
    def __init__(self, k = 5, metric = 'euclidean'):
        self.k = k
        self.metric = metric
    
    def __repr__(self):
        return f"K - Nearest Neighbors Regressor {{ k = {self.k} }}"
    
    def fit(self, X_train, y_train):
        self.X_train_ = X_train
        self.y_train_ = y_train

    def compute_distance(self, x_a, x_b):

        '''
        Computing distances based on the input condition.
        Options: Euclidean, Manhattan.
        '''

        dist = 0

        if self.metric == 'euclidean':
            for a_i, b_i in zip(x_a, x_b):
                dist += (a_i - b_i) ** 2
            dist = dist ** 0.5

        elif self.metric == 'manhattan':    
            for a_i, b_i in zip(x_a, x_b):
                dist += abs(a_i - b_i) 
        
        return dist 

    def get_knn(self, distances):

        '''
        Constructing a list containing the index of single point and the correspondent distance.
        Sorting it and keep the first k points.
        '''

        points = [(distance, i) for i, distance in enumerate(distances)]
        points.sort()   
        k_points = points[:self.k]
        
        return [i for _, i in k_points]
    
    def single_prediction(self, X_point):

        '''
        Get distances from the sample X_point and get k nearest points' indices.
        Perform prediction by returning the mean.
        '''

        distances = [self.compute_distance(X_i, X_point) for X_i in self.X_train_]
        weights = distances/np.sum(distances)

        k_indices = self.get_knn(distances)

        values = [self.y_train_[index] for index in k_indices]
        k_weights = [weights[index] for index in k_indices]

        values = np.multiply(values, k_weights)

        return np.sum(values)
    
    def predict(self, X):
        '''
        Predicting all test samples by iterating single_prediction().
        '''
        return [self.single_prediction(X_j) for X_j in X]

def knnCV (dataset, target, neighbors, folds):

    '''
    Tuning paramaters through cross validation.
    Evaluation metrics: R2 score, (root) mean square deviation, mean absolute error.
    Looping over possible distance metrics and number of NN, passed as input variable.
    Return a matrix for each evaluation metric containing the average score 
    for every choice of parameters.
    '''

    col = 0
    matrix_r2 = np.zeros((len(neighbors), 2))
    matrix_mse = np.zeros((len(neighbors), 2))
    matrix_rmse = np.zeros((len(neighbors), 2))
    matrix_mae = np.zeros((len(neighbors), 2))

    for m in ['euclidean', 'manhattan']:
        print('metric: ', m)
        row = 0
        for j in neighbors:

            r2 = []
            mse = []
            rmse = []
            mae = []
            
            print ('number of neighbors: ' , j)
            
            for i in range(folds):

                X = np.asarray(dataset.drop([target], axis=1).astype(float))
                y = np.asarray(dataset[target].astype(float))

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/folds , random_state = i)

                knn = KNNRegressor(k = j, metric = m)
                # fit
                knn.fit(X_train, y_train)
                y_pred = knn.predict(X_test)

                r2.append(r2_score(y_test, y_pred))
                mse.append(mean_squared_error(y_test, y_pred))
                rmse.append(mean_squared_error(y_test, y_pred)**0.5)
                mae.append(mean_absolute_error(y_test, y_pred))
            
            matrix_r2[row, col] = np.mean(r2)
            matrix_mse[row, col] = np.mean(mse)
            matrix_rmse[row, col] = np.mean(rmse)
            matrix_mae[row, col] = np.mean(mae)

            row += 1
            
        col += 1

    return matrix_r2, matrix_mse, matrix_rmse, matrix_mae

