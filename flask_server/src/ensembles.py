from time import time
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.tree import DecisionTreeRegressor
from tqdm import trange
from metrics import *


class RandomForestMSE:
    def __init__(
            self, n_estimators=100, max_depth=None, feature_subsample_size=0.6,
            **trees_parameters
    ):
        """
        n_estimators : int
            The number of trees in the forest.
        max_depth : int
            The maximum depth of the tree. If None then there is no limits.
        feature_subsample_size : float
            The size of feature set for each tree. If None then use one-third of all features.
        """

        self.n_estimators = n_estimators
        self.feature_subsample_size = feature_subsample_size
        self.max_depth = max_depth
        self.trees_parameters = trees_parameters
        self.estimators = []
        self.feat_subsample = []

    def fit(self, X, y, X_val=None, y_val=None, trace=True):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        y : numpy ndarray
            Array of size n_objects
        X_val : numpy ndarray
            Array of size n_val_objects, n_features
        y_val : numpy ndarray
            Array of size n_val_objects
        """

        history = {'time': [], 'train_score': []}
        start = time()
        train_preds = np.zeros_like(y)
        if y_val is not None:
            history['val_score'] = []
            preds = np.zeros_like(y_val)
        for i in trange(self.n_estimators):
            est = DecisionTreeRegressor(max_depth=self.max_depth,
                                        max_features=self.feature_subsample_size,
                                        **self.trees_parameters)

            idx = np.random.choice(X.shape[0], (X.shape[0],))
            self.feat_subsample.append(np.random.choice(X.shape[1], replace=False,
                                                        size=(int(self.feature_subsample_size * X.shape[1]), )))

            est.fit(X[idx][:, self.feat_subsample[-1]], y[idx])

            train_preds += est.predict(X[idx][:, self.feat_subsample[-1]])
            history['train_score'].append(RMSE(train_preds / (i + 1), y))

            self.estimators.append(est)
            history['time'].append(time() - start)

            if X_val is not None:
                preds += est.predict(X_val[:, self.feat_subsample[-1]])
                history['val_score'].append(RMSE(preds / (i + 1), y_val))

        if trace:
            return history

    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """

        res = np.zeros((X.shape[0],))

        for est, feat in zip(self.estimators, self.feat_subsample):
            res += est.predict(X[:, feat]) / len(self.estimators)

        return res

    def get_n_estimators(self):
        return self.n_estimators

    def get_max_depth(self):
        return self.max_depth

    def get_feature_subsample_size(self):
        return self.feature_subsample_size

    def get_name(self):
        return "RandomForestMSE"


class GradientBoostingMSE:
    def __init__(
            self, n_estimators=100, learning_rate=0.1, max_depth=5, feature_subsample_size=0.6,
            **trees_parameters
    ):
        """
        n_estimators : int
            The number of trees in the forest.
        learning_rate : float
            Use alpha * learning_rate instead of alpha
        max_depth : int
            The maximum depth of the tree. If None then there is no limits.
        feature_subsample_size : float
            The size of feature set for each tree. If None then use one-third of all features.
        """

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.feature_subsample_size = feature_subsample_size

        self.estimators = []
        self.alphas = []
        self.feat_subsample = []
        self.trees_parameters = trees_parameters


    def fit(self, X, y, X_val=None, y_val=None, trace=True):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        y : numpy ndarray
            Array of size n_objects
        """

        history = {'time': [], 'train_score': []}

        curr_f = np.zeros_like(y)
        start = time()
        if y_val is not None:
            history['val_score'] = []
            preds = np.zeros_like(y_val)
        for _ in trange(self.n_estimators):
            est = DecisionTreeRegressor(max_depth=self.max_depth,
                                        **self.trees_parameters)

            antigrad = self.get_antigrad(curr_f, y)
            self.feat_subsample.append(np.random.choice(X.shape[1], replace=False,
									   size=(int(self.feature_subsample_size * X.shape[1]), )))
            est.fit(X[:, self.feat_subsample[-1]], antigrad)
            self.estimators.append(est)

            pred = est.predict(X[:, self.feat_subsample[-1]])
            alpha = minimize_scalar(lambda a: MSE(curr_f + a * pred, y),
                                    bounds=(0, 1000),
                                    method='bounded').x
            self.alphas.append(alpha)

            history['time'].append(time() - start)

            curr_f += self.learning_rate * alpha * pred
            history['train_score'].append(RMSE(curr_f, y))

            if X_val is not None:
                preds += self.learning_rate * alpha * est.predict(X_val[:, self.feat_subsample[-1]])
                history['val_score'].append(RMSE(preds, y_val))

        if trace:
            return history

    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """

        res = np.zeros((X.shape[0],))

        for est, alpha, feat in zip(self.estimators, self.alphas, self.feat_subsample):
            res += self.learning_rate * alpha * est.predict(X[:, feat])

        return res

    def get_antigrad(self, f, y):
        return y - f

    def get_n_estimators(self):
        return self.n_estimators

    def get_max_depth(self):
        return self.max_depth

    def get_feature_subsample_size(self):
        return self.feature_subsample_size

    def get_learning_rate(self):
        return self.learning_rate

    def get_name(self):
        return "GradientBoostingMSE"
