from pathlib import Path
import pandas as pd
import json
import numpy as np

delta = {
    "men": 2*(436)/(252+436) - 1,
    "women": 2*((541)/(407+541)) - 1,
    "both": 2*(541+436)/(407+541+252+436) - 1,
}

def prepare_table_with_results(sex):
    list_of_files = list(Path(".").resolve().parents[0].joinpath("results_xvalidation").glob(f"*_{sex}_*.csv"))

    data_list = []
    for file in list_of_files:
        data = pd.read_csv(file)
        data_list.append(data)
    # Join results from different classifiers
    data_joined = pd.concat(data_list)
    # Sort data in descending order
    data_joined.sort_values("test_mcc", ascending=False, inplace=True)
    # Merge svm_poly and svm_rbf as they should be treated as a single classifier
    data_joined["classifier"] = data_joined["classifier"].apply(lambda x: x.replace("svm_poly", "svm")
                                                                           .replace("svm_rbf", "svm"))
    # Keep only the best result for each classifier type, delete the rest
    data_joined.drop_duplicates(subset="classifier", keep="first", inplace=True)
    # Reorganize to have the same order between sexes
    data_joined.set_index("classifier", inplace=True)
    # data_joined = data_joined.loc[["adaboost", "decisiontree", "gauss_nb", "random_forest", "svm", "knn"]]
    # Rename recall to sensitivity
    data_joined.rename(columns={"test_recall": "test_sensitivity", "test_recall_stdev": "test_sensitivity_stdev"},
                         inplace=True)
    # Sort
    data_joined.sort_values("classifier", inplace=True)

    # drop accuracy
    return data_joined.drop(["test_accuracy", "test_accuracy_stdev"], axis=1)

def generate_tables_for_top_results():
    # Prepare table for computing bias
    bias_latex = (
        "\\begin{table}[h]\n"
        "\\centering\n"
        "\\caption{Estimated mean bias of each metric for best classifier for each sex}\n"
        "\\label{tab:metric_bias}\n"
        "\\begin{tabular}{| l|| r | r || r | r || r |r |}\n"
        "\\hline\n"
        "Metric &  Female & Bias  & Male & Bias & Both & Bias \\\\\n"
        "\\hline\n"
    )
    bias_table = pd.DataFrame(columns=["name", "women_result", "women_bias", "men_result", "men_bias",
                                       "both_result", "both_bias"])
    bias_table.set_index("name", inplace=True)



    # Generate the table with performance
    table_results = prepare_table_with_results("women")
    # Declaration of the latex table
    table_header = ("\\begin{table*}\n"
                    "\\centering\n"
                    f"\\caption{{Best results reached for each classifier type}}\n"
                    f"\\label{{tab:results_top_performance}}\n"
                    f"\\begin{{tabular}}{{ll{"c" * (table_results.shape[1] - 2)}}}"
                    "\\toprule")
    # Ending of the latex table
    table_footer = ("\\bottomrule\n"
                    "\\end{tabular}\n"
                    "\\end{table*}\n")
    # Header of the table itself
    table_body = "\\multirow{2}{*}{Sex} & \\multirow{2}{*}{Classifier} & \\multicolumn{2}{c}{"
    # Header of the table
    # Metrics that will be displayed
    metrics = table_results.columns[["stdev" not in x for x in table_results.columns.tolist()]][2:]
    abbreviations = [metric.split("_")[1].upper()[:3] for metric in metrics]
    table_body += "} & \\multicolumn{2}{c}{".join(abbreviations) + "} \\\\\n"
    # Alternating mean and standard deviation
    table_body += "& " + "& $\\mu$ & $\\sigma$ " * (len(abbreviations)) + "\\\\\n"
    # Middle rule
    table_body += "\\midrule\n"

    for sex in ["women", "men","both"]:
        table = prepare_table_with_results(sex)

        # Compute bias
        # find sensitivity(lpp) and specificity(lnn) for highest mcc
        lpp, lnn = table.loc[table["test_mcc"].idxmax()][["test_sensitivity", "test_specificity"]]
        d = delta[sex]
        # Compute metrics and biases
        bias_table.loc["Sensitivity",f"{sex}_result"] = lpp
        bias_table.loc["Sensitivity",f"{sex}_bias"] = 0
        bias_table.loc["Specificity",f"{sex}_result"] = lnn
        bias_table.loc["Specificity",f"{sex}_bias"] = 0
        bias_table.loc["Accuracy",f"{sex}_result"] = lpp*(1+d)/2 +lnn*(1-d)/2
        bias_table.loc["Accuracy",f"{sex}_bias"] = (d/2)*(lpp-lnn)
        bias_table.loc["F1-score",f"{sex}_result"] = (2*lpp*(1+d))/((1+lpp)*(1+d)+(1-lnn)*(1-d))
        bias_table.loc["F1-score",f"{sex}_bias"] = (2*lpp*(1+d))/((1+lpp)*(1+d)+(1-lnn)*(1-d)) - 2*lpp/(2+lpp-lnn)
        bias_table.loc["Precision",f"{sex}_result"] = (lpp*(1+d))/(lpp*(1+d)+(1-lnn)*(1-d))
        bias_table.loc["Precision",f"{sex}_bias"] = (1+d)/((1+d)+((1-lnn)/lpp)*(1-d)) - 1/(1+(1-lnn)/lpp)
        bias_table.loc["GM",f"{sex}_result"] = np.sqrt(lpp*lnn)
        bias_table.loc["GM",f"{sex}_bias"] = 0
        bias_table.loc["UAR",f"{sex}_result"] = (lpp+lnn)/2
        bias_table.loc["UAR",f"{sex}_bias"] = 0
        bias_table.loc["MCC (normalized)",f"{sex}_result"] = ((lpp+lnn-1)/np.sqrt((lpp+(1-lnn)*(1-d)/(1+d))*(lnn+(1-lnn)*(1+d)/(1-d))))/2 +1/2
        bias_table.loc["MCC (normalized)",f"{sex}_bias"] = (lpp+lnn-1)/(2*np.sqrt((lpp+(1-lnn)*(1-d)/(1+d))*(lnn+(1-lnn)*(1+d)/(1-d)))) - (lpp+lnn-1)/(2*np.sqrt((lpp+1-lnn)*(lnn+1-lpp)))


        # Adding the results
        for idx, (classifier, row) in enumerate(table.iterrows()):
            add = ""
            if idx == 0:
                match sex:
                    case 'women':
                        sex_id = 'F'
                    case 'men':
                        sex_id = 'M'
                    case 'both':
                        sex_id = 'B'
                    case _:
                        raise RuntimeError
                add += f"\\multirow{{{table_results.shape[0]}}}{{*}}{{{sex_id}}} & "
            else:
                add += "& "

            table_row = classifier + " & " + " & ".join([f"{val:.4f}" for val in row.iloc[2:]])
            table_body += add + table_row + "\\\\\n"



        # Middle rule
        table_body += "\\midrule\n"

    # Final touch
    table_body = (table_body.replace("adaboost", "AdaBoost")
                            .replace("decisiontree", "DT")
                            .replace("gauss_nb", "NB")
                            .replace("random_forest", "RF")
                            .replace("svm","SVM")
                            .replace("knn", "KNN"))

    with open("table_top_results.txt", "w") as file:
        file.write(table_header + table_body[:-9] + table_footer)

    # finalise bias table
    for metric, res in bias_table.transpose().to_dict().items():
        bias_latex += f"{metric}"
        for val in res.values():
            bias_latex += f" & {val:.4f}"
        bias_latex += "\\\\\n"
    bias_latex += (
        "\\hline\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )
    with open("bias_table.txt", "w") as file:
        file.write(bias_latex)

def fetch_dataset_information(sex):
    data_list = []
    # Go through each dataset that appears in the best results to find the configuration
    table = prepare_table_with_results(sex)
    for classifier, dataset, config in zip(table.index, table.dataset, table.hyperparameters):
        # Open the config to find the info on the electable features
        path = Path("").resolve().parents[0].joinpath("training_data", sex, str(dataset), "config.json")
        with open(path, "r") as f:
            # Save the info about the sex and about the classifier
            data = pd.DataFrame(json.load(f), index=[f"{sex}_{classifier}"])
            data["dataset"] = dataset
            data["config"] = config
        data_list.append(data)
    # Join all configs together to one table
    data_joined = pd.concat(data_list)
    # Drop obsolete sex
    data_joined.drop(columns=["sex"], inplace=True)
    # Add the compulsory features
    for col in ["f0", "hnr", "jitta", "shim", "age"]:
        data_joined[col] = True
    for col in ["mfcc_delta", "mfcc_2delta"]:
        data_joined[col] = data_joined.mfcc
    # Reform the variance features to have the appropriate values if used
    for col in ["mfcc_var", "mfcc_delta_var", "mfcc_delta2_var"]:
        data_joined[col] = data_joined[["mfcc", "var_mfcc"]].apply(lambda row: row.mfcc if row.var_mfcc else False, axis=1)
    data_joined.drop("var_mfcc", axis=1, inplace=True)
    # Transpose to get the right format
    data_transposed = data_joined.transpose()
    # Change True to "Y" and False to "N"
    data_transposed = data_transposed.map(lambda cell: "Y" if cell is True else cell)
    data_transposed = data_transposed.map(lambda cell: "N" if cell is False else cell)
    # Reorganize rows
    indices = ['config', 'dataset', 'f0', 'hnr', 'jitta', 'shim', 'nan', 'age',
               'stdev_f0', 'diff_pitch', 'shannon_entropy', 'lfcc', 'formants', 'skewness',
               'spectral_centroid', 'spectral_contrast', 'spectral_flatness', 'spectral_rolloff', 'zero_crossing_rate',
               'mfcc', 'mfcc_delta', 'mfcc_2delta',
               'mfcc_var', 'mfcc_delta_var', 'mfcc_delta2_var']
    data_transposed = data_transposed.reindex(indices)
    # Rename rows
    correct_indices = ["config", "dataset",
                       "$\\overline{f}_{0}$","$\\overline{HNR}$", "$jitta$", "$shim$", "$NaN$", "$age$",
                       "$\\sigma_{f_0}$", "$\\Delta f_0$", "$H$", "$\\overline{\\mathbf{LFCC}}$",
                       "$\\overline{\\mathbf{f}}$", "$skew$", "$\\overline{S}$", "$\\overline{SC}$",
                       "$\\overline{SF}$", "$\\overline{RO}$", "$\\overline{ZCR}$",
                       "$\\overline{\\mathbf{MFCC}}$", "$\\overline{\\Delta \\mathbf{MFCC}}$",
                       "$\\overline{\\Delta^2 \\mathbf{MFCC}}$", "$\\mathbf{\\sigma^2_{MFCC}}$",
                       "$\\mathbf{\\sigma^2_{\\Delta MFCC}}$", "$\\mathbf{\\sigma^2_{\\Delta^2MFCC}}$"]
    data_transposed.index = correct_indices
    return data_transposed
def generate_table_with_configurations(sex):
    # Generate the table with performance
    table_results = prepare_table_with_results(sex)
    table_header = ("\\begin{table*}\n"
                    "\\centering\n"
                    f"\\caption{{Configuration of the best model and dataset for each classifier type - {sex}}}\n"
                    f"\\label{{tab:top_results_config_{sex}}}\n"
                    f"\\begin{{tabular}}{{l{"c" * (table_results.index.shape[0])}}}\n"
                    f"\\toprule\n")
    # Ending of the latex table
    table_footer = ("\\bottomrule\n"
                    "\\end{tabular}\n"
                    "\\end{table*}\n")
    # table_body = (f"Sex & \\multicolumn{{{table_results.index.shape[0]}}}{{c}}{{F}} & "
    #               f"\\multicolumn{{{table_results.index.shape[0]}}}{{c}}{{M}} \\\\\n")
    table_body = ("Classifier & " + (" & ".join(table_results.index.tolist())) + "\\\\\n")

    # Adding configurations of the datasets
    table_dataset_config = fetch_dataset_information(sex)
    for index, vals in table_dataset_config.iterrows():
        if index == "dataset":
            table_body += f"Dataset name & " + " & ".join([str(val) for val in vals]) + " \\\\\n"
            table_body += "\\midrule\n"
        elif index == "config":
            formatted_vals = " & \\shortstack[c]".join([str(val) for val in vals])
            formatted_vals = formatted_vals.replace("classifier__", "").replace("_", "\\_")
            formatted_vals = formatted_vals.replace(",", "\\\\")
            table_body += " & \\shortstack[c]" + formatted_vals + "\\\\\n"
            table_body += "\\midrule\n"
        else:
            table_body += f"{index} & " + " & ".join([str(val) for val in vals]) + " \\\\\n"

    # Final touch
    table_body = (table_body.replace("adaboost", "AdaBoost")
                  .replace("decisiontree", "DT")
                  .replace("gauss_nb", "NB")
                  .replace("random_forest", "RF")
                  .replace("svm", "SVM")
                  .replace("knn", "KNN"))

    with open(f"table_top_results_config_{sex}.txt", "w") as file:
        file.write(table_header + table_body + table_footer)

if __name__ == '__main__':
    generate_tables_for_top_results()
    for current_sex in ["men", "women", "both"]:
        generate_table_with_configurations(current_sex)
