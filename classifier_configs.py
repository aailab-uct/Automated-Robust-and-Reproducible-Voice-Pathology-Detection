"""Module for loading pipelines with corresponding classifiers and hyperparameters for grid search."""
from imblearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from src.custom_smote import CustomSMOTE

# definition of classifiers parameters for gridsearch
grids = {
    'svm_poly': {
        "classifier__C": [0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 3000, 5000, 7000, 10000, 12000],
        "classifier__kernel": ["poly"],
        "classifier__gamma": [0.5, 0.1, 0.05, 0.01, 0.005, 0.001, "auto"],
        "classifier__degree": [2, 3, 4, 5, 6]
    },
    'svm_rbf': {
        "classifier__C": [0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 3000, 5000, 7000, 10000, 12000],
        "classifier__kernel": ["rbf"],
        "classifier__gamma": [0.5, 0.1, 0.05, 0.01, 0.005, 0.001, "auto"],
    },
    'knn': {
        "classifier__n_neighbors": [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23],
        "classifier__weights": ["uniform", "distance"],
        "classifier__p": [1, 2]},
    'gauss_nb': {
        "classifier__var_smoothing": [1e-9, 1e-8]},
    'random_forest': {
        "classifier__n_estimators": [50, 75, 100, 125, 150, 175],
        "classifier__criterion": ["gini"],
        "classifier__min_samples_split": [2, 3, 4, 5, 6],
        "classifier__max_features": ["sqrt"]
    },
    'adaboost': {
        "classifier__n_estimators": [50, 100, 150, 200, 250, 300, 350, 400],
        "classifier__learning_rate": [0.1, 1, 10]
    },
    'decisiontree': {
        "classifier__criterion": ["gini", "log_loss", "entropy"],
        "classifier__min_samples_split": [2, 3, 4, 5, 6, 7, 8, 9, 10],
        "classifier__splitter": ["best", "random"],
        "classifier__max_features": ["log2", "sqrt"],
    }

}


def get_classifier(classifier_name: str, random_seed: int = 42, hyperparameters: dict = None):
    """
    Get classifier with the given name.
    :param classifier_name: str, name of the classifier
    currently supported: "svm_poly", "svm_rbf", "knn", "gauss_nb", "random_forest", "adaboost"
    :param random_seed: int, random seed
    :param hyperparameters: dict, hyperparameters if already known (for repeated cross-validation)
    default: None
    :return: classifier
    :return: grid for grid search
    """
    if hyperparameters is None:
        processed_hyperparameters = {}
    else:
        processed_hyperparameters = {}
        for key, value in hyperparameters.items():
            key = key.replace("classifier__", "")
            processed_hyperparameters[key] = value

    if classifier_name == "svm_poly":
        pipe = Pipeline([
            ("smote", CustomSMOTE(random_state=random_seed)),
            ("minmaxscaler", MinMaxScaler()),
            ("classifier", SVC(max_iter=int(1e6),
                               random_state=random_seed,
                               **processed_hyperparameters))
        ])
    elif classifier_name == "svm_rbf":
        pipe = Pipeline([
            ("smote", CustomSMOTE(random_state=random_seed)),
            ("minmaxscaler", MinMaxScaler()),
            ("classifier", SVC(max_iter=int(1e6),
                               random_state=random_seed,
                               **processed_hyperparameters))
        ])
    elif classifier_name == "knn":
        pipe = Pipeline([
            ("smote", CustomSMOTE(random_state=random_seed)),
            ("minmaxscaler", MinMaxScaler()),
            ("classifier", KNeighborsClassifier(**processed_hyperparameters))
        ])
    elif classifier_name == "gauss_nb":
        pipe = Pipeline([
            ("smote", CustomSMOTE(random_state=random_seed)),
            ("minmaxscaler", MinMaxScaler()),
            ("classifier", GaussianNB(**processed_hyperparameters))
        ])
    elif classifier_name == "random_forest":
        pipe = Pipeline([
            ("smote", CustomSMOTE(random_state=random_seed)),
            ("minmaxscaler", MinMaxScaler()),
            ("classifier", RandomForestClassifier(random_state=random_seed,
                                                  **processed_hyperparameters))
        ])
    elif classifier_name == "adaboost":
        pipe = Pipeline([
            ("smote", CustomSMOTE(random_state=random_seed)),
            ("minmaxscaler", MinMaxScaler()),
            ("classifier", AdaBoostClassifier(random_state=random_seed,
                                              algorithm="SAMME",
                                              **processed_hyperparameters))
        ])
    elif classifier_name == "decisiontree":
        pipe = Pipeline([
            ("smote", CustomSMOTE(random_state=random_seed)),
            ("minmaxscaler", MinMaxScaler()),
            ("classifier", DecisionTreeClassifier(random_state=random_seed,
                                                  **processed_hyperparameters))
        ])
    else:
        raise ValueError(f"Unknown classifier: {classifier_name}")
    return pipe, grids[classifier_name]
