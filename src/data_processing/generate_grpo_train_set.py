import json
import random
from pathlib import Path
from argparse import ArgumentParser

from transformers import AutoTokenizer

from data_processing.ludii_utils import load_ludii_example, dict_to_xml


def create_dataset(args):
    with open(args.test_game_path, "r") as f:
        test_game_dict = json.load(f)
    test_games = list(test_game_dict.keys())
    gamelist = []
    if Path(args.base_dir_path_or_gamelist).is_file():
        with open(args.base_dir_path_or_gamelist, "r") as f:
            gamelist = f.read().splitlines()
    else:
        for game in Path(args.base_dir_path_or_gamelist).rglob("*.lud"):
            if "experimental" not in str(game):
                gamepath = str(game).split(args.base_dir_path_or_gamelist)[-1]
                gamelist.append(gamepath)
    train_games = [game for game in gamelist if game not in test_games]
    test_games = random.sample(test_games, 5)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    Path(args.output_dataset_dir).mkdir(parents=True, exist_ok=True)
    train_path = Path(args.output_dataset_dir) / "train.json"
    test_path = Path(args.output_dataset_dir) / "test.json"

    train_dataset = []
    train_gamename_list = []
    for gamepath in train_games:
        example = load_ludii_example(gamepath)
        program_token_length = tokenizer(example.target, return_tensors="pt")[
            "input_ids"
        ].shape[1]
        if program_token_length > args.max_token_len or not example.target.startswith(
            "(game"
        ):
            continue
        source = dict_to_xml({"task": {"query": example.source}})
        train_dataset.append({"problem": source, "solution": example.target})
        train_gamename_list.append(gamepath)
    print(f"Created train dataset with {len(train_dataset)} examples")
    with open(train_path, "w") as f:
        json.dump(train_dataset, f, indent=4)
    with open(Path(args.output_dataset_dir) / "gamelist.txt", "w") as f:
        f.write("\n".join(train_gamename_list))

    test_dataset = []
    for gamepath in test_games:
        example = load_ludii_example(gamepath)
        program_token_length = tokenizer(example.target, return_tensors="pt")[
            "input_ids"
        ].shape[1]
        source = dict_to_xml({"task": {"query": example.source}})
        test_dataset.append({"problem": source, "solution": example.target})
    print(f"Created test dataset with {len(test_dataset)} examples")
    with open(test_path, "w") as f:
        json.dump(test_dataset, f, indent=4)


if __name__ == "__main__":
    args = ArgumentParser()

    # data/ludii/expand/ or data/ludii/gamelist_concepts.txt
    args.add_argument(
        "--base_dir_path_or_gamelist",
        type=str,
        default="data/ludii/gamelist_concepts.txt",
    )

    args.add_argument("--output_dataset_dir", type=str, default="data/ludii/grpo/")
    args.add_argument(
        "--test_game_path", type=str, default="data/ludii/eval/default.json"
    )
    args.add_argument("--tokenizer", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    args.add_argument("--max_token_len", type=int, default=500)
    args = args.parse_args()

    create_dataset(args)
