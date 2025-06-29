import argparse
import subprocess
import multiprocessing
from pathlib import Path
from functools import partial

from tqdm import tqdm

from data_processing.ludii_utils import load_ludii_example


TEST_FILENAME_LIST = "data/ludii/gamelist.txt"
CONCEPT_FILENAME_LIST = "data/ludii/gamelist_concepts.txt"
CONCEPTS_DIR = "data/ludii/concepts"


def extract_concept(test_filename, num_threads):
    print(f"{test_filename}")

    example = load_ludii_example(test_filename)
    game_fullname = example.gamepath.split(".")[0]
    game_fullpath = Path("data/ludii/Ludii/Common/res/lud") / example.gamepath
    trial_dir = Path("data/ludii/trials_") / game_fullname
    concept_dir = Path("data/ludii/concepts_") / game_fullname

    result = subprocess.run(
        [
            "java",
            "-jar",
            "ludii_java/jars/ComputeConcept.jar",
            "--trials-dir",
            str(trial_dir),
            "--concepts-dir",
            str(concept_dir),
            "--game-path",
            str(game_fullpath),
            "--num-threads",
            str(num_threads),
            "--num-trials",
            "100",
        ],
        capture_output=True,
        text=True,
    )
    return result


def get_all_json_files(directory):
    path = Path(directory)
    json_files = list(path.rglob("*.json"))
    json_files = [str(file.absolute()) for file in json_files]
    return json_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate concepts for Ludii games.")
    parser.add_argument(
        "--num_processes", type=int, default=12, help="Number of processes to use."
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=50,
        help="Number of threads for each concept extraction.",
    )
    args = parser.parse_args()

    with open(TEST_FILENAME_LIST, "r") as file:
        test_filenames = file.read().splitlines()

    target_filenames = [
        test_filename
        for test_filename in test_filenames
        if not (
            Path(CONCEPTS_DIR) / test_filename.replace(".lud", "") / "Concepts.csv"
        ).exists()
    ]

    print(f"Games: {len(target_filenames)}")

    partial_extract_concept = partial(extract_concept, num_threads=args.num_threads)

    if args.num_processes == 1:
        for target_filename in tqdm(target_filenames):
            result = partial_extract_concept(target_filename)
            print(result.stderr)
    else:
        with multiprocessing.Pool(processes=args.num_processes) as pool:
            results = pool.map(partial_extract_concept, target_filenames)

    print("Generation is done!")

    print("Check if all concepts are generated")
    generated_filenames = []
    for filename in test_filenames:
        if (
            Path(CONCEPTS_DIR) / filename.replace(".lud", "") / "Concepts.csv"
        ).exists():
            generated_filenames.append(filename)
    print(f"Generated: {len(generated_filenames)}")

    with open(CONCEPT_FILENAME_LIST, "w") as file:
        for filename in generated_filenames:
            file.write(f"{filename}\n")
