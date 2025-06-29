import json
import random
from collections import defaultdict
from pathlib import Path
from argparse import ArgumentParser

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

from data_processing.ludii_utils import load_ludii_example


MUST_INCLUDE_CATEGORIES = [
    "board/race",
    "board/sow",
    "puzzle",
    "board/space/line",
    "board/war",
]


def create_dataset(args):
    gamelist = []
    if Path(args.base_dir_path_or_gamelist).is_file():
        with open(args.base_dir_path_or_gamelist, "r") as f:
            gamelist = f.read().splitlines()
    else:
        for game in Path(args.base_dir_path_or_gamelist).rglob("*.lud"):
            if "experimental" not in str(game):
                gamepath = str(game).split(args.base_dir_path_or_gamelist)[-1]
                gamelist.append(gamepath)

    demo_in_icl_gamelist = []
    with open(args.demo_in_icl_gamelist, "r") as f:
        demo_in_icl_gamelist = f.read().splitlines()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    example_by_group = defaultdict(list)
    for game in gamelist:
        example = load_ludii_example(game)
        target_token_length = tokenizer(example.target, return_tensors="pt")[
            "input_ids"
        ].shape[1]
        if (
            target_token_length > args.max_token_len
            or target_token_length < args.min_token_len
            or not example.target.startswith("(game")
        ):
            continue
        group = str(Path(example.gamepath).parent)
        if args.group != "" and not group.startswith(args.group):
            continue
        example_by_group[group].append(example)

    test_game_dict = {}
    if args.is_different_category:
        all_groups = list(example_by_group.keys())
        for group_name, test_example_group in tqdm(list(example_by_group.items())):
            other_group_games = []
            for other_group_name in all_groups:
                if other_group_name != group_name:
                    other_group_games.extend(example_by_group[other_group_name])
            for e_id, test_example in enumerate(test_example_group):
                demo_examples = np.random.choice(
                    other_group_games, args.num_demo_examples, replace=False
                )
                test_game_dict[test_example.gamepath] = [
                    demo_example.gamepath for demo_example in demo_examples
                ]
    else:
        for group_name, test_example_group in tqdm(list(example_by_group.items())):
            if len(test_example_group) < args.num_demo_examples + 1:
                continue
            for e_id, test_example in enumerate(test_example_group):
                others = np.delete(test_example_group, e_id)
                others = [
                    other for other in others if other.gamepath in demo_in_icl_gamelist
                ]
                demo_examples = np.random.choice(
                    others, args.num_demo_examples, replace=False
                )
                test_game_dict[test_example.gamepath] = [
                    demo_example.gamepath for demo_example in demo_examples
                ]

    use_game_test_examples = list(test_game_dict.keys())
    if len(use_game_test_examples) < args.num_games:
        random_test_game_dict = test_game_dict
    else:
        random_test_game_dict = {}
        for must_include_category in MUST_INCLUDE_CATEGORIES:
            must_include_category_games = [
                game for game in use_game_test_examples if must_include_category in game
            ]
            num_samples = min(
                args.num_games_per_category, int(len(must_include_category_games) * 0.5)
            )
            use_must_include_category_games = random.sample(
                must_include_category_games, num_samples
            )
            for game in use_must_include_category_games:
                random_test_game_dict[game] = test_game_dict[game]
                use_game_test_examples.remove(game)
        random_keys = random.sample(
            use_game_test_examples, args.num_games - len(random_test_game_dict)
        )
        for key in random_keys:
            random_test_game_dict[key] = test_game_dict[key]

    random_test_game_by_group = defaultdict(list)
    for game, demo_games in random_test_game_dict.items():
        group = str(Path(game).parent)
        random_test_game_by_group[group].append(game)

    print(f"Number of games: {len(random_test_game_dict)}")
    for must_include_category in MUST_INCLUDE_CATEGORIES:
        games_in_category = [
            game
            for game in random_test_game_dict.keys()
            if must_include_category in game
        ]
        print(f"{must_include_category}: {len(games_in_category)}")

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w") as file:
        json.dump(random_test_game_dict, file, indent=4)


if __name__ == "__main__":
    args = ArgumentParser()

    # data/ludii/expand/ or data/ludii/gamelist_concepts.txt
    args.add_argument(
        "--base_dir_path_or_gamelist",
        type=str,
        default="data/ludii/gamelist_concepts.txt",
    )
    args.add_argument(
        "--demo_in_icl_gamelist", type=str, default="data/ludii/gamelist_grammar.txt"
    )
    args.add_argument("--output_path", type=str, default="data/ludii/eval/default.json")

    args.add_argument("--tokenizer", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    args.add_argument("--max_token_len", type=int, default=500)
    args.add_argument("--min_token_len", type=int, default=0)

    args.add_argument("--num_games", type=int, default=100)
    args.add_argument("--num_demo_examples", type=int, default=3)

    args.add_argument("--num_games_per_category", type=int, default=10)
    args.add_argument("--group", type=str, default="")
    args.add_argument("--is_different_category", action="store_true")
    args = args.parse_args()

    create_dataset(args)
