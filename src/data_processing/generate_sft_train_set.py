import json
import random
from pathlib import Path
from argparse import ArgumentParser
from textwrap import dedent

from transformers import AutoTokenizer

from data_processing.ludii_utils import load_ludii_example, dict_to_xml


def create_dataset(args):
    # Load prompts
    with open(args.prompt_path, "r") as f:
        prompts = json.load(f)
    system_program_prompt = prompts["system_prompt_template"][
        "zero-shot_program_generation"
    ]

    with open(args.test_game_path, "r") as f:
        test_game_dict = json.load(f)
    test_games = list(test_game_dict.keys())

    if args.train_gamelist_path:
        with open(args.train_gamelist_path, "r") as f:
            gamelist = f.read().splitlines()
        train_games = [game for game in gamelist if game not in test_games]
        assert len(train_games) == len(gamelist), (
            f"{list(set(gamelist) - set(train_games))}"
        )
    else:
        gamelist = []
        for game in Path(args.base_dir_path).rglob("*.lud"):
            if "experimental" not in str(game):
                gamepath = str(game).split(args.base_dir_path)[-1]
                gamelist.append(gamepath)
        train_games = [game for game in gamelist if game not in test_games]

    test_games = random.sample(test_games, 5)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    Path(args.output_dataset_dir).mkdir(parents=True, exist_ok=True)
    train_path = Path(args.output_dataset_dir) / "train.json"
    test_path = Path(args.output_dataset_dir) / "test.json"

    program_token_len_list = []
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
        program_token_len_list.append(program_token_length)

        user_content = dict_to_xml({"task": {"query": example.source}})
        assistant_content = (
            dict_to_xml({"program": dedent(example.target)}) + tokenizer.eos_token
        )
        program_data = {
            "messages": [
                {"role": "system", "content": system_program_prompt},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ]
        }
        train_dataset.append(program_data)
        train_gamename_list.append(gamepath)
    random.shuffle(train_dataset)
    print(f"Created dataset with {len(train_dataset)} examples")
    print(f"Program Token Len: {sum(program_token_len_list)}")
    with open(train_path, "w") as f:
        json.dump(train_dataset, f, indent=4)
    with open(Path(args.output_dataset_dir) / "gamelist.txt", "w") as f:
        f.write("\n".join(train_gamename_list))

    test_dataset = []
    for gamepath in test_games:
        example = load_ludii_example(gamepath)
        program_data = {
            "messages": [
                {"role": "system", "content": system_program_prompt},
                {
                    "role": "user",
                    "content": dict_to_xml({"task": {"query": example.source}}),
                },
                {
                    "role": "assistant",
                    "content": dict_to_xml({"program": dedent(example.target)}),
                },
            ]
        }
        test_dataset.append(program_data)
    print(f"Created test dataset with {len(test_dataset)} examples")
    with open(test_path, "w") as f:
        json.dump(test_dataset, f, indent=4)


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--base_dir_path", type=str, default="data/ludii/expand/")
    args.add_argument(
        "--prompt_path", type=str, default="data/ludii/prompts/default.json"
    )
    args.add_argument("--output_dataset_dir", type=str, default="data/ludii/sft/")
    args.add_argument(
        "--test_game_path", type=str, default="data/ludii/eval/default.json"
    )
    args.add_argument(
        "--train_gamelist_path", type=str, default="data/ludii/grpo/gamelist.txt"
    )
    args.add_argument("--tokenizer", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    args.add_argument("--max_token_len", type=int, default=500)
    args = args.parse_args()

    create_dataset(args)
