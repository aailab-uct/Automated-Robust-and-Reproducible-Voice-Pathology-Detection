# Automated Robust and Reproducible Voice Pathology Detection
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13771573.svg)](https://doi.org/10.5281/zenodo.13771573)

This repository contains the code for the paper "Towards the Automated Robust and Reproducible Voice
Pathology Detection: A Machine Learning Approach" by Jan Vrba, Jakub Steinbach, Tomáš Jirsa, Laura Verde, Roberta De Fazio, Noriyasu Homma, Yuwen Zeng, Kei Ichiji, Lukáš Hájek, Zuzana Sedláková, Jan Mareš

## Requirements
For running experiments
- prepared dataset (see below)
- reproducing the results is computationally demanding, thus we suggest to use as many CPUs as possible

Used libraries and software
- Python 3.12.4
- see requirements.txt for all dependencies

Used setup for experiments
- AMD Ryzen 5900X 12-cores CPU
- 128 GB RAM
- 2 TB SSD disk
- Ubuntu 22.04 LTS

## Notes on reproducibility

We tried to make the results as reproducible as possible. Although we employed many approaches to ensure the reproducibility, it is still possible that the results will not be exactly the same, mainly due to the floating point arithmetic. We provide the checksums for all files that are generated during the experiments. The checksums are stored in the `misc` folder. The checksums are computed using the code in `src/checksums`, however, they can be computed using 7-zip or sha256sum utility.

When running the `main.py` script, the script checks the checksums of the downloaded data, the extracted features, the generated datasets, the results of the classifiers, the average results, and the results of the repeated stratified cross-validation.

**Beware** that git on Windows can change the line endings from LF to CRLF. This can cause the checksums to fail. Most probably you only need to change EOL of `misc\svd_information.csv`. Additionally, check [this link](https://docs.github.com/en/get-started/getting-started-with-git/configuring-git-to-handle-line-endings) on how to configure git to handle this problem. The script should save every csv file with LF as EOL.

Additionally, we provide `our_results_tables` and `our_results_xvalidation` folders that contain the results of our experiments. We do not include the `results` folder as it is too large to be stored in the repository, even when compressed.


## Dataset preparation
The dataset is not included in this repository due to the license reason, but it can be downloaded from publicly
available website. Firstly download the Saarbruecken Voice Database (SVD)
[available here](https://stimmdb.coli.uni-saarland.de/help_en.php4). You need to download all recordings of /a/ vowel
produced at normal pitch that are encoded as wav files. Due to the limitation of SVD interface, download male recordings
and female recordings separately. Then create the `svd_db` folder in the root of this project and put all recordings
there.
At this step we assume following folder structure:
```
svm_voices_research
└───misc
└───src
└───svd_db
    │   1-a_n.wav
    │   2-a_n.wav
    │   ...
    │   2610-a_n.wav
```

We provide the `svd_information.csv` file that contains the information about the SVD database (age, sex, pathologies, etc.). The file is stored in the `misc` folder and contains data scraped from the SVD website.

When running the `main.py` script, the script checks the checksums of the downloaded data to ensure that the dataset is complete and consistent.

## Reproducing the results

After successfully downloading the whole SVD our results can be reproduced by running the `main.py` script, executing
the following command

```python main.py ```

Additionally, you can specify the number of threads used for dataset generation by setting the `MAX_WORKERS` variable in the `main.py` script. The default value is 24. Moreover, you can run individual steps of the pipeline by running the corresponding script. These scripts usually allow to specify sex and classifier type you want to run. However, the `main.py` script is the most convenient way to reproduce the results, provides checking of the checksums of the generated files, and was tested to work as expected.

This script automatically performs following tasks from the corresponding scripts:

0. **checking downloaded data consistency**

During this step, it is checked if your downloaded files from SVD correspond with our dataset.
This is achieved via comparison of the checksum of your data with the checksum in the `misc/data_usd.sha256`.

1. `data_preprocessing.py` - **removing the corrupted files and duplicities and triming silence**

We remove corrupted files and duplicities from `svd_db` and create a `datasets` folder that contains SVD database wav files with trimmed silences. The resulting `datasets` is checked using the checksum stored in a `misc/after_I.sha256`.

2. `feature_extraction.py` - **extracting features from preprocessed data**

We employ many feature extraction methods to extract features from the preprocessed data as described in the article.
The resulting `features.csv` file should have the same checksum as the checksum stored in the `misc/after_II.sha256` file.

3. `dataset_generator.py` - **generating all datasets**

The resulting `training_data` folder should contain two subfolders `training_data/men` and `training_data/women`. Each of those subfolders should contain 8192 subfolders with pickled dataset `dataset.pk` and its
configuration stored in `configuration.json`. The checksum of resulting dataset folders is checked and should be same as the provided checksum in `misc/after_III.sha256` file. The dataset generating can be sped up by setting the number of threads in `MAX_WORKERS` variable (see line 20 in `main.py`).

4. `classifier_pipeline.py` - **running the machine learning pipeline**.

This step is computationally extremely demanding and it took more than 14 days to finish on the recommended setup. **The number of cores (or CPUs) can
significantly reduce the computation time!**. The result is the `results` folder containing subfolders `men` and `women` that contains the subfolders with results for all datasets generated in previous step. The results are saved into the `results.csv` files. All files with results are checked across the checksums provided in `misc/after_IV.sha256` file.

5. `average_results.py` - **computing average results**

In this step the computation of average results across all parameters for each classifier and each dataset is done.
The resulting folder `results_tables` should contain csv files for each classifier and sex. The csv file contains
the average metrics for each dataset (average across all hyperparameters evaluated during the grid search performed
in the previous step 5.). The resulting files are checked across the checksums provided in `misc/after_V.sha256` file.

6. `repeated_cross_validation_best.py` - **repeated stratified cross-validation of the best classifiers**

This step is computationally demanding. For each type of classifier, we select 1000 the best classifiers
(configuration of hyperparameters and dataset) and perform repeated (100 times) stratified 10-fold cross-validation.
The results are saved in the `results_xvalidation` folder, that contains subfolders named according to classifiers names.
In each of those subfolders is a `xvalidation_results.csv` file with the performance of the classifiers during the
above mentioned process. The results of this step are checked across the checksums provided
in `misc/after_VI.sha256` file.


## Description of files in this repository:

- **main.py** - script to reproduce all our results
- **requirements.txt** - list of used packages
- **analyze_results.py** - script to check the best results produced by `classifier_pipeline.py`
- **average_results.py** - script to compute the average metrics for single dataset (run after *_pipeline.py) is finished
- **classifier_configs.py** - configuration of classifiers pipelines with corresponding hyperparameters specification
for the grid search
- **classifier_pipeline.py** - grid search for classifiers specified in `classifier_configs.py` on all generated
datasets
- **data_preprocessing.py** - script that removes corrupted files and duplicities
- **dataset_generator.py** - script that generates datasets from preprocessed data
- **feature_extraction.py** - script that extracts features from preprocessed data
- **README.md** - this file
- **repeated_cross_validation_best.py** - script that performs repeated stratified cross-validation of the best classifiers
- **src** - folder with additional code
    - **checksum.py** - script that computes checksums of the files
    - **custom_metrics.py** - script with custom metrics used in the classifiers
    - **custom_smote.py** - script with custom SMOTE implementation
- **misc** - folder with additional files
    - **after_I.sha256** - checksum of the `datasets` folder after the `data_preprocessing.py` script
    - **after_II.sha256** - checksum of the `features.csv` file after the `feature_extraction.py` script
    - **after_III.sha256** - checksum of the `training_data` folder after the `dataset_generator.py` script
    - **after_IV.sha256** - checksum of the `results` folder after the `classifier_pipeline.py` script
    - **after_V.sha256** - checksum of the `results_tables` folder after the `average_results.py` script
    - **after_VI.sha256** - checksum of the `results_xvalidation` folder after the `repeated_cross_validation_best.py` script
    - **data_used.sha256** - checksum of the downloaded data
    - **list_of_excluded_files.csv** - list of excluded files from the SVD database with the reason of exclusion
    - **svd_information.csv** - information about the SVD database
- **article_tables** - folder with code to generate tables used in the article
    - **generate_best_dataset_result_table.py** - script that handles generation of the latex table for the best performing datasets for each classifier and sex
    - **generate_tables_for_top_results.py** - script for generating a latex table with the best results and their configurations
    - **table_best_datasets_template.txt** - latex template
- **our_results_tables** - our results after running the `average_results.py` script
- **our_results_xvalidation** - our results after running the `repeated_cross_validation_best.py` script
