"""
Script that performs grid search for all classifiers across various datasets.
Datasets are balanced via KMeansSMOTE and results are obtained via stratified
 10-fold cross-validation.
"""
import argparse
import pickle
from pathlib import Path
from typing import Union

import tqdm
import numpy as np
import pandas as pd

from imblearn.metrics import geometric_mean_score
import sklearn
from sklearn.model_selection import GridSearchCV,StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score, matthews_corrcoef, balanced_accuracy_score
from sklearn.metrics import make_scorer

from classifier_configs import get_classifier
from src.checksum import update_checksums

# Disable warnings
import os, warnings
warnings.simplefilter("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# setup random seed to make code reproducible
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# speedup
sklearn.set_config(
    assume_finite=False,
    skip_parameter_validation=True,
)

# specification of evaluated metrics
scoring_dict = {"mcc": make_scorer(matthews_corrcoef),
                "accuracy": make_scorer(accuracy_score),
                "recall": make_scorer(recall_score),
                "specificity": make_scorer(recall_score, pos_label=0),
                "gm": make_scorer(geometric_mean_score),
                "uar": make_scorer(balanced_accuracy_score, adjusted=False),
                "bm": make_scorer(balanced_accuracy_score, adjusted=True)}


def get_datasets_to_process(datasets_path: Path, results_data: Path,
                            dataset_slice: Union[tuple, int, None] = None):
    """
    Get datasets that were not evaluated.
    :param datasets_path: Path, path to training datasets
    :param results_data: Path, path to results folder (used to check which datasets were already evaluated)
    :param dataset_slice: int, slice of datasets to evaluate. If None all datasets will be evaluated.
    If tuple, slice will be used, e.g. (10, 100). If int, first n datasets will be evaluated.
    :return: list of datasets to evaluate
    """
    # get all datasets in training_data
    td = sorted([str(x.name) for x in datasets_path.iterdir()])
    if isinstance(dataset_slice, tuple):
        td = list(td)[dataset_slice[0]:dataset_slice[1]]
    elif isinstance(dataset_slice, int):
        td = list(td)[:dataset_slice]

    # get all datasets, that were already evaluated
    tr = sorted([str(x.name) for x in results_data.iterdir()])
    # to perform gridsearch only for datasets that were not evaluated so far
    to_do = sorted(list(set(td) - set(tr)))

    return to_do

# pylint: disable=too-many-locals


def main(sex: str = "women", classifier = "svm_poly", dataset_slice = None):
    """
    Main function for the classifier pipeline.

    :param sex: The sex for which the classifier is trained. Default is "women".
    :param classifier: The type of classifier to use. Default is "svm_poly".
    :param dataset_slice: The slice of the dataset to process. Default is None.
    """
    # setup paths to training data and results
    training_data = Path(".").joinpath("training_data", sex)
    results_data = Path(".").joinpath("results", classifier, sex)
    results_data.mkdir(exist_ok=True, parents=True)

    # get datasets to be evaluated
    dataset = get_datasets_to_process(training_data, results_data, dataset_slice)
    dataset = sorted(dataset)

    for training_dataset_str in tqdm.tqdm(dataset):
        results_file = results_data.joinpath(str(training_dataset_str))
        # path to training dataset
        training_dataset = training_data.joinpath(training_dataset_str)
        #print(f"evaluate {training_dataset}")
        results_data.joinpath(str(training_dataset.name)).mkdir(parents=True, exist_ok=True)
        # load dataset
        with open(training_data.joinpath(str(training_dataset.name), "dataset.pk"), "rb") as f:
            train_set = pickle.load(f)
        dataset = {"X": np.array(train_set["data"], dtype=np.float64),
                   "y": np.array(train_set["labels"])}
        # imblearn pipeline perform the resampling only with the training dataset
        # and scaling according to training dataset
        pipeline, param_grid = get_classifier(classifier, both_sexes=sex=="both", random_seed=RANDOM_SEED)
        cross_validation = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)
        # sklearn gridsearch with cross validation
        grid_search = GridSearchCV(pipeline, param_grid, cv=cross_validation, scoring=scoring_dict,
                                   n_jobs=-1, refit=False)
        grid_search.fit(dataset["X"], dataset["y"])
        # create folder with the training_dataset name to store results
        results_file.mkdir(exist_ok=True)
        # no need to write header again and again and again,...
        if results_file.joinpath("results.csv").exists():
            header = False
        else:
            header = True

        # dump gridsearch results to a results.csv
        pd.DataFrame(grid_search.cv_results_).round(6)[
            ["params", "mean_test_accuracy", "mean_test_recall", "mean_test_specificity",
             "mean_test_mcc", "mean_test_gm", "mean_test_uar", "mean_test_bm"]].to_csv(
            results_file.joinpath("results.csv"),
            index=False, mode="a",
            header=header,encoding="utf8",lineterminator="\n")
# pylint: enable=too-many-locals


if __name__ == "__main__":
    # argument parser
    parser = argparse.ArgumentParser(
        prog="classifier_pipeline.py",
        description="Perform grid search for different classifier across various datasets. "
    )
    parser.add_argument("classifier", type=str)
    parser.add_argument("sex", type=str)
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    sex_to_compute = args.sex
    used_classifier = args.classifier
    data_slice = None if not args.test else 50
    main(sex_to_compute, used_classifier, data_slice)
    # compute results checksum and compare it with ours results to ensure the reproducible results
    update_checksums(Path("results"),Path("misc").joinpath("after_IV.sha256"))
