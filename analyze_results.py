"""Analyzes the results of the experiments."""
from pathlib import Path
import json
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

MIN_ACC = 0.8  # 0.8577
MIN_RECALL = 0.8  # 0.8759
MIN_SPEC = 0.8  # 0.8394
results_dir = Path("results")
temp_df_list = []
for path in sorted(results_dir.glob("**/results.csv")):
    df = pd.read_csv(path)
    df['uar'] = df.apply(lambda row: (row.mean_test_recall + row.mean_test_specificity) / 2, axis=1)

    temp_df_list.append(df)
    temp_df_list[-1]["data"] = path.parent.name

table_results = pd.concat(temp_df_list)
table_results = table_results[(table_results.mean_test_accuracy >= MIN_ACC) &
                              (table_results.mean_test_recall >= MIN_RECALL) &
                              (table_results.mean_test_specificity >= MIN_SPEC)].sort_values("mean_test_mcc",
                                                                                             kind="mergesort")

for _, row in table_results.tail(10).iterrows():
    str_params = ""
    for key, val in json.loads(row["params"].replace("'", '"')).items():
        if key == "classifier__C":
            str_params += f"{key.split('__')[1]}: {val:>5} - "
        elif key == "classifier__gamma":
            if val == "auto":
                str_params += f"{key.split('__')[1]}: {val:>5} - "
            else:
                str_params += f"{key.split('__')[1]}: {val:.3f} - "
        elif key == "classifier__kernel":
            str_params += f"{key.split('__')[1]}: {val:>4} - "
        else:
            str_params += f"{key.split('__')[1]}: {val} - "

    print(
        f'{int(row["data"]):>4} | {str_params[:-2]}| Acc: {row["mean_test_accuracy"]:.5f}'
        f' - Sen: {row["mean_test_recall"]:.5f}'
        f' - Spe: {row["mean_test_specificity"]:.5f} - UAR: {row["uar"]:.5f} - MCC: {row["mean_test_mcc"]:.5f}')

print("")
print(table_results["data"].value_counts().reset_index().head(30))
table_results["data"].value_counts().reset_index().plot.hist()
table_results["data"].value_counts().reset_index().plot.bar()
plt.show()
