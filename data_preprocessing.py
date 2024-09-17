"""
Script that loads the dataset, preprocess it to desired format, drops undesired files,
and removes silence from the recordings.
"""
from pathlib import Path

import librosa
import soundfile as sf
import pandas as pd
from tqdm import tqdm


# Path to the dataset folder where all recordings should be saved
PATH_SVD = Path('svd_db')
# Path to the information file where the information on all recordings from the SVD database are saved
PATH_FILE_INFORMATION = Path(".").joinpath('misc','svd_information.csv')
# Output directory for the trimmed files
PATH_DATASET = Path(".").joinpath("dataset")


def trim_silence(input_path, output_path):
    """
    Function that removes the silence from the WAV file specified by input_path.
    :param input_path: path to file that will be trimmed.
    :param output_path: path to output file, where the trimmed file will be written.
    :return: None
    """
    # Load the audio file
    y, sr = librosa.load(input_path, sr=None)

    # Trim silence from the beginning and end
    yt, _ = librosa.effects.trim(y, top_db=15, frame_length=256, hop_length=65)

    # Save the trimmed audio file
    sf.write(output_path, yt, sr)

def drop_files(file_information_table, age_dict, silent=False, removed_files_path: Path = None):
    """
    Function that drops files from the table based on the conditions specified in the paper.
    The conditions are:
    - Drop corrupted files
    - Drop data of young patients
    - Drop multiple recordings of the same person
    :param file_information_table: table with the information on the recordings
    :param age_dict: dictionary with the age limits for each sex
    :param silent: boolean flag to suppress the print statements
    :param removed_files_path: path to the file listing the removed files and the reason for their removal, if None,
    the file will not be created
    :return: table with the dropped files
    """
    removed = pd.DataFrame(columns=["session_id", "reason", "age", "talker_id", "pathology"])
    # Drop corrupted files
    session_id_to_delete = [1573, 87]
    idx_to_drop = file_information_table[file_information_table.session_id.isin(session_id_to_delete)].index
    removed = pd.concat(
        [removed,
        file_information_table.loc[idx_to_drop, ["session_id", "age", "talker_id",
                                                 "pathology"]].assign(reason="corrupted")])
    file_information_table = file_information_table.drop(idx_to_drop)


    if not silent:
        print("Without corrupted:\n\t",file_information_table.shape)

    # Iterate through the items of the dictionary and find the indices which are the correct combination of sex and
    # below age limit. Then, delete the rows which are represented by these indices.
    for sex, age_limit in age_dict.items():
        idx_to_drop = file_information_table[(file_information_table.sex == sex) &
                                             (file_information_table.age < age_limit)].index
        removed = pd.concat(
            [removed,
            file_information_table.loc[idx_to_drop, ["session_id", "age", "talker_id",
                                                     "pathology"]].assign(reason="age")])
        file_information_table.drop(idx_to_drop, inplace=True)

    if not silent:
        print("After age restriction:\n\t",file_information_table.shape)

    # Sort the values based on the session date. There are multiple recordings from some of the subjects. To prevent
    # potential data leakage, we will take only the oldest recordings of the healthy and pathological sessions for each
    # person. That means each person has maximum of 2 recordings in the final dataset, one being diagnosed
    # as pathologic, one as healthy. Resetting the index prevents pandas "remembering" it in case it chooses the index
    # to decide the order of occurrence for dropping the duplicates. To make the order of occurrence deterministic, we
    # additionally sort the recordings by the session id to ensure deterministic order of occurrence if the session date
    # is the same.
    file_information_table = file_information_table.sort_values(["session_date", "session_id"], ascending=True) \
                                                   .reset_index(drop=True)
    # Drop all duplicates of talker id and diagnosis combinations with keeping only the first record
    file_information_table_unique = file_information_table.drop_duplicates(subset=["talker_id", "pathology_binary"],
                                                                           keep="first")

    removed = pd.concat(
        [removed,
        file_information_table.loc[file_information_table.index.difference(file_information_table_unique.index),
                                   ["session_id", "age", "talker_id",
                                   "pathology"]].assign(reason="multiple recordings")])

    if not silent:
        print("After dropping multiple recording of same person:\n\t",file_information_table_unique.shape)

    if removed_files_path is not None:
        removed = removed.assign(pathology=removed["pathology"].fillna("healthy"))
        removed.to_csv(removed_files_path, index=False, lineterminator='\n')
    return file_information_table_unique


def main(source_path: Path = PATH_SVD,
         destination_path: Path = PATH_DATASET,
         file_information_path: Path = PATH_FILE_INFORMATION,
         removed_files_path: Path = None):
    """
    Main function that trims the silence from the recordings in the SVD database and saves them to the destination_path.
    :param source_path: path to the folder with the original recordings
    :param destination_path: path to the folder where the trimmed recordings should be saved
    :param file_information_path: path to the file with the information on the recordings
    :param removed_files_path: path to the file listing the removed files and the reason for their removal, if None,
    the file will not be created
    :return: None
    """
    # Load the file information file, specifically the session id (name of the recording),
    # session date, talker id (id of the patient), and pathology
    patient_information_table = (
        pd.read_csv(file_information_path)[["sessionid", "sessiondate", "talkerid", "pathologies",
                                            "talkersex", "talkerage"]]
        .rename(columns={"sessionid": "session_id", "sessiondate": "session_date", "talkersex": "sex",
                         "talkerage": "age", "talkerid": "talker_id", "pathologies": "pathology"}))

    # List of all recordings which should be stored as WAVE files
    list_of_files = list(source_path.glob('[0-9]*-a_n.wav'))

    # Table which contains the file paths and information on the files
    file_information_table = pd.DataFrame(data=list_of_files, columns=["path"])
    # Save the session id from the file name
    file_information_table["session_id"] = file_information_table["path"].apply(lambda x: int(x.stem.split("-")[0]))

    # Merge the patient and file information together
    file_information_table = file_information_table.merge(patient_information_table, on="session_id", how="left")
    # Change the diagnosis information to respect the binary nature of the classification
    file_information_table["pathology_binary"] = file_information_table["pathology"].apply(
        lambda x: 0 if pd.isna(x) else 1)
    # Change the sex to integer values
    file_information_table["sex"] = file_information_table["sex"].apply(
        lambda x: 0 if x == "m" else 1)

    print("All files:\n\t", file_information_table.shape)

    # Drop data of young patients - we set the limit to 18+ years for both sex
    age_dict = {
        0: 18,  # for men (labeled by 0)
        1: 18   # for women (labeled by 1)
    }
    # Drop corrupted files, drop data of young patients, drop multiple recordings of the same person
    file_information_table_unique = drop_files(file_information_table, age_dict,
                                               removed_files_path=removed_files_path)

    # Create output directory if it doesn't exist
    destination_path.mkdir(parents=True)

    # Iterate over selected recordings and process .wav files to remove the silent parts
    for file in tqdm(file_information_table_unique.path.values):
        trimmed_file_path = destination_path.joinpath(file.stem + "_trim.wav")
        if not trimmed_file_path.exists():
            # remove the silence
            trim_silence(file, trimmed_file_path)
        else:
            print(f"File {trimmed_file_path.name} already exists, skipping.")

    # Ensure paths are represented as strings in posix format
    file_information_table_unique["path"] = file_information_table_unique["path"].apply(lambda x: x.as_posix())
    # Save the information about the selected trimmed files as CSV
    file_information_table_unique.to_csv(destination_path.joinpath("dataset_information.csv"),
                                         index=False, lineterminator='\n')


if __name__ == "__main__":
    main(removed_files_path=Path("misc").joinpath("list_of_excluded_files.csv"))
