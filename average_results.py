"""
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!         RUN THIS SCRIPT ONLY WHEN CLASSIFICATIONS ARE FINISHED        !!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
Script to compute average performance of the various classifiers with various
hyperparameters for each dataset. It iterates over results folder and
corresponding subfolders and evaluates all results.csv files.
"""
from pathlib import Path
from typing import List
import tqdm
import pandas as pd

from src.checksum import update_checksums

# Disable warnings
import os, warnings
warnings.simplefilter("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

def compute_stats(file_path):
    """
    Function to compute mean performance metrics value (+ standard deviations)
    for a single dataset.
    :param file_path: path to results.csv
    :return: dictionary with average metrics and corresponding standard deviations
    """
    # Read the CSV file
    df = pd.read_csv(file_path)
    # Exclude the first column with classifier settings
    df_excluded = df.iloc[:, 1:]
    # Compute mean and standard deviation for each column
    means = df_excluded.mean()
    std_devs = df_excluded.std()
    # prepare dictionary with results
    results = {"name": file_path.parents[0].name}
    for col in df_excluded.columns:
        results[f"{col}_mean"] = [means[col]]
        results[f"{col}_std"] = [std_devs[col]]
    return results


def process_results(classifiers: List[str]):
    """
    Function to iterate over results/svm folder and evaluate
    the average performance of various classifiers with various hyperparameters
    on each dataset.
    :param classifiers: list of classifiers to evaluate
    :return: None
    """
    # path to directory with tables
    results_tables_dir = "results_tables"
    # create the directory for tables
    Path(".", results_tables_dir).mkdir(parents=True, exist_ok=True)
    # iterate over classifiers to evaluate avg. performance of all classifiers
    for classifier in classifiers:
        # path to results for a classifier
        path_to_results = Path(".", "results", classifier)

        for folder in sorted(path_to_results.iterdir()):
            # iterate over men/women/both
            result_summary = Path(results_tables_dir,
                                  f"results_{classifier}_{folder.name}.csv")
            if result_summary.is_file():
                print(f"File {result_summary} already exists. Overwriting...")

            all_stats = pd.DataFrame()
            print(f"Processing {classifier} results for {folder.name}....")
            for result_dir in tqdm.tqdm(sorted(folder.iterdir())):
                # iterate over datasets and corresponding results.csv
                result_file = result_dir.joinpath("results.csv")
                # compute average metrics
                exp_stats = compute_stats(result_file)

                all_stats = pd.concat([all_stats, pd.DataFrame(exp_stats)], ignore_index=True)

            # dump average metrics for a dataset into csv file
            all_stats.to_csv(result_summary, header=True, index=False, mode="w",
                             sep=",", encoding="utf-8", lineterminator="\n")


if __name__ == "__main__":
    process_results(classifiers=["svm_poly", "svm_rbf", "knn", "random_forest", "gauss_nb", "adaboost", "decisiontree"])
    update_checksums(Path("results_tables"), Path("misc").joinpath("after_V.sha256"))
