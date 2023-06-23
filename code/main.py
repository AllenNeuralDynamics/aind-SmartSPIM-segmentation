"""
Main file to execute the smartspim segmentation
in code ocean
"""

import cProfile
import os
import pstats
import subprocess
import sys

from src.aind_smartspim_segmentation import segmentation


def save_string_to_txt(txt: str, filepath: str, mode="w") -> None:
    """
    Saves a text in a file in the given mode.

    Parameters
    ------------------------
    txt: str
        String to be saved.

    filepath: PathLike
        Path where the file is located or will be saved.

    mode: str
        File open mode.

    """

    with open(filepath, mode) as file:
        file.write(txt + "\n")


def execute_command_helper(command: str, print_command: bool = False) -> None:
    """
    Execute a shell command.

    Parameters
    ------------------------
    command: str
        Command that we want to execute.
    print_command: bool
        Bool that dictates if we print the command in the console.

    Raises
    ------------------------
    CalledProcessError:
        if the command could not be executed (Returned non-zero status).

    """

    if print_command:
        print(command)

    popen = subprocess.Popen(command, stdout=subprocess.PIPE, universal_newlines=True, shell=True)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield str(stdout_line).strip()
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def main():
    """
    Main function to execute the smartspim segmentation
    in code ocean
    """

    results_folder = os.path.abspath("../results")
    profiler = cProfile.Profile()
    profiler.enable()
    image_path = segmentation.main()
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.dump_stats(f"{results_folder}/segmentation_stats.stats")
    profiler.dump_stats(f"{results_folder}/segmentation_profile.prof")

    bucket_path = "aind-open-data"

    dataset_folder = str(sys.argv[4]).split("/")[2]
    channel_name = image_path.split("/")[-2].replace(".zarr", "")

    dataset_name = dataset_folder + f"/image_cell_segmentation/{channel_name}"
    s3_path = f"s3://{bucket_path}/{dataset_name}"

    # for out in execute_command_helper(f"aws s3 mv --recursive {results_folder} {s3_path}"):
    #    print(out)

    save_string_to_txt(
        f"Results of cell segmentation saved in: {s3_path}",
        f"{results_folder}/output_segmentation.txt",
    )


if __name__ == "__main__":
    main()
