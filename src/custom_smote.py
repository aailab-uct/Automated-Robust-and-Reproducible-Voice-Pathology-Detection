"""
This module contains a custom implementation of SMOTE algorithm. The custom implementation
is based on the KMeansSMOTE algorithm from the imbalanced-learn library with fallback to
standard SMOTE in case KMeansSMOTE fails 10 times.
"""
import numpy as np
from imblearn.base import BaseSampler
from imblearn.over_sampling import SMOTE, KMeansSMOTE


class CustomSMOTE(BaseSampler):
    """Class that implements KMeansSMOTE oversampling. Due to initialization of KMeans
    there are 10 tries to resample the dataset. Then standard SMOTE is applied.
    """
    _sampling_type = "over-sampling"

    def __init__(self, per_sex:bool = False, random_state=None, kmeans_args=None, smote_args=None):
        super().__init__()
        self.random_state = random_state
        self.kmeans_args = kmeans_args if kmeans_args is not None else {}
        self.smote_args = smote_args if smote_args is not None else {}
        if random_state is not None:
            self.kmeans_args["random_state"] = random_state
            self.smote_args["random_state"] = random_state
        self.kmeans_smote = KMeansSMOTE(**self.kmeans_args)
        self.smote = SMOTE(**self.smote_args)
        self.per_sex = per_sex

    def _fit_resample(self, X, y):
        """
        Resampling dataset to make it balanced.
        :param X: features
        :param y: labels
        :return: resampled features and corresponding labels
        """
        resample_try = 0
        # sometimes k-means SMOTE may fail, so we perform it 10x. It is very unlikely, that the failure will occur
        # more than once
        while resample_try < 10:
            try:
                if self.per_sex:
                    X_male = X[X[:, 0] == 0]
                    y_male = y[X[:, 0] == 0]
                    X_female = X[X[:, 0] == 1]
                    y_female = y[X[:, 0] == 1]
                    X_male, y_male = self.kmeans_smote.fit_resample(X_male,y_male)
                    X_female, y_female = self.kmeans_smote.fit_resample(X_female,y_female)
                    X_res = np.concatenate([X_female,X_male], axis=0)
                    y_res = np.concatenate([y_female,y_male], axis=0)
                else:
                    X_res, y_res = self.kmeans_smote.fit_resample(X, y)
                return X_res, y_res
            # pylint: disable=broad-exception-caught
            except Exception:
                # dont care about exception, KmeansSMOTE failed
                if "random_state" in self.kmeans_args:
                    # change the random seed to change the initial cluster positions and prevent k-means SMOTE fail
                    self.kmeans_args["random_state"] += 1
                self.kmeans_smote = KMeansSMOTE(**self.kmeans_args)
                resample_try += 1
        if self.per_sex:
            X_male = X[X[:, 0] == 0]
            y_male = y[X[:, 0] == 0]
            X_female = X[X[:, 0] == 1]
            y_female = y[X[:, 0] == 1]
            X_female, y_female = self.smote.fit_resample(X_female,y_female)
            X_male, y_male = self.smote.fit_resample(X_male,y_male)
            X_res = np.concatenate([X_female,X_male], axis=0)
            y_res = np.concatenate([y_female,y_male], axis=0)
        else:
            X_res, y_res = self.smote.fit_resample(X, y)
        return X_res, y_res
