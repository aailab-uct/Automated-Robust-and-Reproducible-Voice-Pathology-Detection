"""
This script handles generation of the latex table for the best performing datasets for each classifier and sex
"""

import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json


SEX = "women"  # Change this to "women" if you want to generate a table for women

def best_dataset_for_each_clf(sex):
    """
    Creating a latex table from all results for a single gender to report the best performing datasets for each classifier type
    param sex: men or women, selection of what kind of results should be reported
    return result_merged: Table with both the best results for each classifier and the dataset configuration
    """
    # Path to the results file - iterating through all results files, loading them as pandas dataframe and saving them into a list
    dir_results = Path().resolve().parents[0].joinpath("results")
    list_path_results = list(dir_results.glob(f"**/{sex}/*/results.csv"))
    list_tables = []
    print("Loading all results...")
    for path in tqdm(list_path_results):
        list_tables.append(pd.read_csv(path))
        list_tables[-1]["classifier"] = path.parent.parent.parent.name
        list_tables[-1]["dataset"] = path.parent.name

    # Concatenating all tables into one large table
    table_overview = pd.concat(list_tables, ignore_index=True)
    table_overview["classifier"] = table_overview["classifier"].apply(lambda x: "svm" if "svm" in x else x)

    # Grouping the data based on the classifier type - dataset pairs to get the average of each metric along with the standard deviation of each metric
    table_avg_results = table_overview.set_index(["classifier", "dataset"]).drop(["params", "mean_test_accuracy"],
                                                                                 axis=1).groupby(
        ["classifier", "dataset"]).agg(["mean", "std"])
    # Agg creates multiindex columns, they need to be flattened
    table_avg_results.columns = ["_".join(a) for a in table_avg_results.columns.to_flat_index()]
    # Finding the index of the max value for each classifier
    idx = table_avg_results.groupby('classifier')['mean_test_mcc_mean'].idxmax()
    # Filtering the table with results to get only one result for each classifier (which is the best in average MCC)
    result = table_avg_results.loc[idx].reset_index()

    # Reorganizing the columns to fit the template
    result = result[['classifier', 'dataset',
                     'mean_test_mcc_mean', 'mean_test_mcc_std',
                     'mean_test_recall_mean', 'mean_test_recall_std',
                     'mean_test_specificity_mean', 'mean_test_specificity_std',
                     'mean_test_uar_mean', 'mean_test_uar_std',
                     'mean_test_gm_mean', 'mean_test_gm_std',
                     'mean_test_bm_mean', 'mean_test_bm_std']]

    # Saving the dataset numbers to find their feature configurations
    data_to_find = result.dataset.values
    # Path to the training data configurations
    path_to_datasets = Path(".").resolve().parents[0].joinpath("training_data", SEX)
    # Setting up a table where the configurations will be loaded
    feature_table = pd.DataFrame({"x": ["f0", "HNR", "jitter", "shimmer", "NaN", "age", "stdev f0", "diff pitch",
                                        "entropy", "lfcc", "formants", "skewness",
                                        "centroid", "contrast", "flatness", "rolloff", "ZCR", "mfcc", "delta mfcc",
                                        "delta2 mfcc", "var mfcc", "var delta mfcc",
                                        "var delta2 mfcc", ]})
    # Iterating through each dataset configuration and saving it into the table
    print("Adding dataset configuration to each result...")
    for experiment_name in data_to_find:
        json_path = path_to_datasets.joinpath(experiment_name, "config.json")

        try:
            with open(json_path) as f:
                used_features = json.load(f)
                # First six features are always used, we include them for more clarity
                feature_table[f"{experiment_name[:5]}"] = ["Y", "Y", "Y", "Y",
                                                           used_features["nan"],
                                                           "Y",
                                                           used_features["stdev_f0"],
                                                           used_features["diff_pitch"],
                                                           used_features["shannon_entropy"],
                                                           used_features["lfcc"],
                                                           used_features["formants"],
                                                           used_features["skewness"],
                                                           used_features["spectral_centroid"],
                                                           used_features["spectral_contrast"],
                                                           used_features["spectral_flatness"],
                                                           used_features["spectral_rolloff"],
                                                           used_features["zero_crossing_rate"],
                                                           # MFCC and their derivative - always the same number
                                                           used_features["mfcc"],
                                                           used_features["mfcc"],
                                                           used_features["mfcc"],
                                                           # Variance of MFCC and their derivatives - either not used or the same number as MFCC
                                                           used_features["mfcc"] if used_features["var_mfcc"] else "N",
                                                           used_features["mfcc"] if used_features["var_mfcc"] else "N",
                                                           used_features["mfcc"] if used_features["var_mfcc"] else "N",

                                                           ]
        except Exception as ex:
            print(ex)
            pass
    # Replacing True/False with "Y"/"N"
    feature_table = feature_table.replace(True, "Y")
    feature_table = feature_table.replace(False, "N")
    # Transposition in order to merge the table with the result table
    feature_table = feature_table.set_index("x").transpose()

    # Merging the result and feature_table and transposing it again, to fit the final template
    result_merged = result.merge(feature_table, how="outer", left_on="dataset",
                                 right_on=feature_table.index).transpose()

    # Substituing the now empty column names with the classifiers to switch their position
    result_merged.columns = result_merged.loc["classifier"]
    result_merged = result_merged[["svm", "knn", "gauss_nb", "decisiontree", "random_forest", "adaboost"]]

    # Dropping the classifier and dataset rows as they are now obsolete
    result_merged.drop(["classifier", "dataset"], axis=0, inplace=True)

    return result_merged

def generate_table_with_results(table):
    """
    Generating a table with all the results for the article for a single gender to report the best performing datasets for each classifier
    param table: Table with all the results for a single gender to report the best performing datasets for each classifier type
    return None
    """
    rows = []
    # Concatenate values in each row with & so it can be added to the latex table template
    print("Generating latex table...")
    for _, row in table.iterrows():
        rows.append(" & ".join([f"{x:.4f}" if isinstance(x, float) else str(x) for x in row.values]))

    latex_table = ""
    counter = 0
    # Adding the concatenated lines to the template to fill in the values
    with open("table_best_datasets_template.txt", "r") as file:
        for line in file:
            if line.endswith("&\n"):
                latex_table += line[:-1] + " " + rows[counter] + " \\\\ \n"
                counter += 1
            else:
                latex_table += line
    with open(f"latex_table_best_datasets_{SEX}.txt", "w") as file:
        file.write(latex_table)
    print("Done!")


if __name__ == "__main__":
    table = best_dataset_for_each_clf(SEX)
    generate_table_with_results(table)
