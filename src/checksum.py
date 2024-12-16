"""
This module contains functions to calculate the checksum of a file and to check if the checksums
in a list match the checksum of the corresponding files.
"""
import hashlib
from pathlib import Path
import tqdm
import pandas as pd

def checksum(file_path):
    """
    Calculate the checksum of a file.
    :param file_path: The Path object of the file.
    :return: The checksum of the file.
    """
    hash_object = hashlib.sha256()
    # Open the file in binary mode
    with open(file_path, "rb") as file:
        # Read the file in chunks
        for chunk in iter(lambda: file.read(4096), b""):
            # Update the hash object with each chunk
            hash_object.update(chunk)
    # Return the hexadecimal digest of the hash
    return hash_object.hexdigest()

def test_line_from_file(line: str, ignore_missing_files=False, quiet=False):
    """
    Check if the checksum in the line matches the checksum of the corresponding file.
    :param line: The line containing the checksum and the file name.
    :param ignore_missing_files: If True, ignore missing files. If False, print a message if a file is missing.
    :param quiet: If True, do not print messages for missing files.
    :return: True if the checksum matches, else False.
    """
    # Split the line into file name and checksum
    checksum_value, check_path = line.strip().split("  ")
    # Create a Path object for the file
    check_path = Path(check_path)
    # Calculate the checksum of the file
    if check_path.exists():
        file_not_found = False
        file_checksum = checksum(check_path)
        result = file_checksum == checksum_value
    elif ignore_missing_files:
        file_not_found = False
        result = True
    else:
        file_not_found = True
        result = False

    if not result and not file_not_found and not quiet:
        print(f"\nChecksum mismatch for file: {check_path}")
    elif file_not_found and not quiet:
        print(f"\nFile not found: {check_path}")
    return result

def sort_checksums(filename: str):
    """
    Sort checksums in a file.
    :param filename: str, path to a file with checksums
    """
    # read file
    df = pd.read_csv(filename, sep=";", header=None)
    # split checksums
    df = df[0].str.split("  ", expand=True)
    # sort checksums
    df = df.sort_values(by=1)
    # write file
    with open(filename, "w") as f:
        for i in range(len(df)):
            f.write(f"{df.iloc[i, 0]}  {df.iloc[i, 1]}\n")

def test_file_list_from_file(file_path: Path, quiet=False, quick=False, ignore_missing_files=False):
    """
    Check if the checksums in the list match the checksum of the corresponding files.
    :param file_path: The Path object of the file containing the checksums.
    :param quiet: If True, do not print messages for missing files.
    :param quick: If True, return as soon as a checksum mismatch is found.
    :return: True if all checksums match, else False.
    :return: A list of tuples containing the file name and the result of the checksum comparison.
    """
    if not file_path.exists():
        return []
    results = []
    # Open the file in read mode
    with open(file_path, "r", encoding="utf8") as file:
        lines = file.readlines()

    failed_files = 0
    if not quiet:
        t = tqdm.tqdm(lines)
    else:
        t = lines

    for line in t:
        result = test_line_from_file(line, quiet=quiet,
                                     ignore_missing_files=ignore_missing_files)
        if not result:
            if quick:
                return False, results
            failed_files += 1

    if not quiet:
        print("All files checked.")
        print(f"Failed: {failed_files}/{len(lines)}")
    return failed_files==0, results

def update_checksums(directory_path, checksum_path, quiet=False):
    """
    Update the checksums in the file with the checksums of new files.
    :param directory_path: The Path object of the directory containing the files.
    :param checksum_path: The Path object of the file containing the checksums.
    :param quiet: If True, do not print messages for missing files.
    """
    checked = []
    if checksum_path.exists():
        # Open the file in read mode
        with open(checksum_path, "r", encoding="utf8") as file:
            lines = file.readlines()
        for line in lines:
            # Split the line into file name and checksum
            _, check_path = line.strip().split("  ")
            # Create a Path object for the file
            check_path = Path(check_path)
            checked.append(check_path)

    if not quiet:
        print("Adding checksums of new results.")
    directory = Path(directory_path)
    results_files = set(list(directory.rglob("*.csv")))
    new_files = results_files - set(checked)
    with open(checksum_path, "w", encoding="utf8") as file:
        if not quiet:
            t = tqdm.tqdm(new_files)
        else:
            t = new_files
        for file_name in t:
            file.write(f"{checksum(file_name)}  {file_name.as_posix()}\n")
    if not quiet:
        print(f"Added checksums of {len(new_files)} new results.")
        print("Please don't forget to share the new checksums with the team.")
        print(f"The new checksums were appended to {checksum_path}")
        print("Sorting the checksums.")
    sort_checksums(checksum_path)
    if not quiet:
        print("Checksums sorted.")
