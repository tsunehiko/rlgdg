"""Reward functions for GRPO training."""

import math
import re
import subprocess
import json
import multiprocessing
import uuid
from functools import partial
from typing import Dict
from pathlib import Path
from math import isnan

import pandas as pd

from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""

    rewards = []
    for completion in completions:
        s = completion[0]["content"]
        start_tag = "<program>"
        end_tag = "</program>"

        start_index = s.find(start_tag)
        if start_index == -1:
            rewards.append(0.0)
            continue
        
        end_index = s.find(end_tag, start_index + len(start_tag))
        if end_index == -1:
            rewards.append(0.0)
            continue
        
        before = s[:start_index]
        after = s[end_index + len(end_tag) :]

        reward = (len(s) - len(before) - len(after)) / len(s)
        rewards.append(reward)
    return rewards


def reasoning_steps_reward(completions, **kwargs):
    r"""Reward function that checks for clear step-by-step reasoning.
    Regex pattern:
        Step \d+: - matches "Step 1:", "Step 2:", etc.
        ^\d+\. - matches numbered lists like "1.", "2.", etc. at start of line
        \n- - matches bullet points with hyphens
        \n\* - matches bullet points with asterisks
        First,|Second,|Next,|Finally, - matches transition words
    """
    pattern = r"(Step \d+:|^\d+\.|\n-|\n\*|First,|Second,|Next,|Finally,)"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [len(re.findall(pattern, content)) for content in completion_contents]

    # Magic nubmer 3 to encourage 3 steps and more, otherwise partial reward
    return [min(1.0, count / 3) for count in matches]


def len_reward(
    completions: list[Dict[str, str]], solutions: list[str], **kwargs
) -> float:
    """Compute length-based rewards to discourage overthinking and promote token efficiency.

    Taken from from the Kimi 1.5 tech report: https://arxiv.org/abs/2501.12599

    Args:
        completions: List of model completions
        solutions: List of ground truth solutions

    Returns:
        List of rewards where:
        - For correct answers: reward = 0.5 - (len - min_len)/(max_len - min_len)
        - For incorrect answers: reward = min(0, 0.5 - (len - min_len)/(max_len - min_len))
    """
    contents = [completion[0]["content"] for completion in completions]

    # First check correctness of answers
    correctness = []
    for content, sol in zip(contents, solutions):
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        if len(gold_parsed) == 0:
            # Skip unparseable examples
            correctness.append(True)  # Treat as correct to avoid penalizing
            print("Failed to parse gold solution: ", sol)
            continue

        answer_parsed = parse(
            content,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        equations=True,
                        boxed=True,
                        units=True,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )
        correctness.append(verify(answer_parsed, gold_parsed))

    # Calculate lengths
    lengths = [len(content) for content in contents]
    min_len = min(lengths)
    max_len = max(lengths)

    # If all responses have the same length, return zero rewards
    if max_len == min_len:
        return [0.0] * len(completions)

    rewards = []
    for length, is_correct in zip(lengths, correctness):
        lambda_val = 0.5 - (length - min_len) / (max_len - min_len)

        if is_correct:
            reward = lambda_val
        else:
            reward = min(0, lambda_val)

        rewards.append(float(reward))

    return rewards


def get_cosine_scaled_reward(
    min_value_wrong: float = -1.0,
    max_value_wrong: float = -0.5,
    min_value_correct: float = 0.5,
    max_value_correct: float = 1.0,
    max_len: int = 1000,
):
    def cosine_scaled_reward(completions, solution, **kwargs):
        """Reward function that scales based on completion length using a cosine schedule.

        Shorter correct solutions are rewarded more than longer ones.
        Longer incorrect solutions are penalized less than shorter ones.

        Args:
            completions: List of model completions
            solution: List of ground truth solutions

        This function is parameterized by the following arguments:
            min_value_wrong: Minimum reward for wrong answers
            max_value_wrong: Maximum reward for wrong answers
            min_value_correct: Minimum reward for correct answers
            max_value_correct: Maximum reward for correct answers
            max_len: Maximum length for scaling
        """
        contents = [completion[0]["content"] for completion in completions]
        rewards = []

        for content, sol in zip(contents, solution):
            gold_parsed = parse(
                sol,
                extraction_mode="first_match",
                extraction_config=[LatexExtractionConfig()],
            )
            if len(gold_parsed) == 0:
                rewards.append(1.0)  # Skip unparseable examples
                print("Failed to parse gold solution: ", sol)
                continue

            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed=True,
                            units=True,
                        ),
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )

            is_correct = verify(answer_parsed, gold_parsed)
            gen_len = len(content)

            # Apply cosine scaling based on length
            progress = gen_len / max_len
            cosine = math.cos(progress * math.pi)

            if is_correct:
                min_value = min_value_correct
                max_value = max_value_correct
            else:
                # Swap min/max for incorrect answers
                min_value = max_value_wrong
                max_value = min_value_wrong

            reward = min_value + 0.5 * (max_value - min_value) * (1.0 + cosine)
            rewards.append(float(reward))

        return rewards

    return cosine_scaled_reward


def extract_program(text):
    pattern_full = re.compile(r"<program>(.*?)</program>", re.DOTALL | re.MULTILINE)
    pattern_partial = re.compile(r"^<program>(.*)$", re.DOTALL | re.MULTILINE)
    pred_programs = [match for match in pattern_full.findall(text)]
    if len(pred_programs) == 0:
        pred_programs = [match for match in pattern_partial.findall(text)]
    if len(pred_programs) == 0:
        return ""
    else:
        return pred_programs[0].strip()


def extract_gamename(program):
    game_match = re.search(r'\(game\s+"([^"]+)"', program)
    if game_match:
        return game_match.group(1)
    else:
        match_match = re.search(r'\(match\s+"([^"]+)"', program)
        if match_match:
            return match_match.group(1)
        else:
            return str(uuid.uuid4())


def get_column_value(df: pd.DataFrame, column_name: str, row_id: int = 0):
    if column_name in df.columns:
        if isnan(df[column_name][row_id]):
            return None
        else:
            return df[column_name][row_id]
    else:
        return None


def gaussian_kernel(x, mu, sigma=0.3):
    return 1.0 - math.exp(-0.5 * ((x - mu) / sigma) ** 2)


def process_concept_reward(input, gamename_dict, concept_gt_dir, function_weight=1.0):
    pred_program, trial_dir, concept_dir, pred_dir, gamename, player_count = input

    # sum of weights should be 0.9
    if player_count > 1:
        decision_weight = 0.18
        coverage_weight = 0.18
        timeout_weight = 0.18
        balance_weight = 0.18
        decisiveness_weight = 0.18
    else:
        decision_weight = 0.3
        coverage_weight = 0.3
        timeout_weight = 0.3
        balance_weight = 0.0
        decisiveness_weight = 0.0

    try:
        result = subprocess.run(
            [
                "java",
                "-jar",
                "ludii_java/jars/EvalLudiiGame.jar",
                "--game",
                pred_program,
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )  # 1 min timeout
        result_str_list = re.findall(r"\{'isCompilable':.*?\}", result.stdout)
        if len(result_str_list) == 0:
            result_dict = {
                "isCompilable": "false",
                "isFunctional": "false",
                "isPlayable": "false",
            }
        else:
            result_dict = json.loads(result_str_list[0].strip().replace("'", '"'))
    except Exception:
        result_dict = {
            "isCompilable": "false",
            "isFunctional": "false",
            "isPlayable": "false",
        }

    if result_dict["isFunctional"] != "true":
        return 0.0

    reward = 1.0 * function_weight
    try:
        result = subprocess.run(
            [
                "java",
                "-jar",
                "ludii_java/jars/ComputeConcept.jar",
                "--trials-dir",
                trial_dir,
                "--concepts-dir",
                concept_dir,
                "--game-path",
                pred_dir,
                "--num-threads",
                "10",
                "--num-trials",
                "10",
                "--short",
                "true",
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )
    except Exception:
        # Timeout or other error -> penalize
        return reward * 0.1

    content_concept_csv = concept_dir / "Concepts.csv"
    if content_concept_csv.exists():
        pred_df = pd.read_csv(content_concept_csv)
        gamepath = gamename_dict[gamename]
        gt_df = pd.read_csv(Path(concept_gt_dir) / gamepath / "Concepts.csv")
        penalty_dict = {}

        pred_agency = get_column_value(pred_df, "DecisionMoves")
        gt_agency = get_column_value(gt_df, "DecisionMoves")
        if pred_agency is not None and gt_agency is not None:
            decision_penalty = gaussian_kernel(pred_agency, gt_agency) * decision_weight
            penalty_dict["DecisionMoves"] = decision_penalty

        pred_coverage = get_column_value(pred_df, "BoardCoverageUsed")
        gt_coverage = get_column_value(gt_df, "BoardCoverageUsed")
        if pred_coverage is not None and gt_coverage is not None:
            coverage_penalty = (
                gaussian_kernel(pred_coverage, gt_coverage) * coverage_weight
            )
            penalty_dict["BoardCoverageUsed"] = coverage_penalty

        pred_timeouts = get_column_value(pred_df, "Timeouts")
        gt_timeouts = get_column_value(gt_df, "Timeouts")
        if pred_timeouts is not None and gt_timeouts is not None:
            timeouts_penalty = (
                gaussian_kernel(1.0 - pred_timeouts, 1.0 - gt_timeouts) * timeout_weight
            )
            penalty_dict["Timeouts"] = timeouts_penalty

        if player_count > 1:
            pred_balance = get_column_value(pred_df, "Balance")
            gt_balance = get_column_value(gt_df, "Balance")
            if pred_balance is not None and gt_balance is not None:
                balance_penalty = (
                    gaussian_kernel(pred_balance, gt_balance) * balance_weight
                )
                penalty_dict["Balance"] = balance_penalty

            pred_decisiveness = get_column_value(pred_df, "Completion")
            gt_decisiveness = get_column_value(gt_df, "Completion")
            if pred_decisiveness is not None and gt_decisiveness is not None:
                decisiveness_penalty = (
                    gaussian_kernel(pred_decisiveness, gt_decisiveness)
                    * decisiveness_weight
                )
                penalty_dict["Completion"] = decisiveness_penalty

        reward -= sum(penalty_dict.values())
    else:
        # No concept file generated -> penalize
        reward = reward * 0.1

    return reward


def get_concept_reward(
    trial_dir, concept_dir, pred_dir, gamename_dict, function_weight=1.0
):
    CONCEPTS_GT_DIR = "data/ludii/concepts"

    def concept_reward(completions, solution, **kwargs):
        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        epoch_dict = {}

        process_concept_reward_func = partial(
            process_concept_reward,
            gamename_dict=gamename_dict,
            concept_gt_dir=CONCEPTS_GT_DIR,
            function_weight=function_weight,
        )

        input_list = []
        for id, (content, sol) in enumerate(zip(contents, solution)):
            gamename = extract_gamename(sol)
            game_trial_dir = Path(trial_dir) / gamename
            game_concept_dir = Path(concept_dir) / gamename
            game_pred_dir = Path(pred_dir) / gamename
            if gamename not in epoch_dict:
                if game_trial_dir.exists():
                    exist_epochs = [
                        int(d.name.split("_")[-1])
                        for d in game_trial_dir.iterdir()
                        if d.is_dir()
                    ]
                    if len(exist_epochs) > 0:
                        epoch_dict[gamename] = max(exist_epochs) + 1
                else:
                    epoch_dict[gamename] = 0
            game_trial_dir = game_trial_dir / f"epoch_{epoch_dict[gamename]}"
            game_concept_dir = game_concept_dir / f"epoch_{epoch_dict[gamename]}"
            game_pred_dir = game_pred_dir / f"epoch_{epoch_dict[gamename]}"
            game_pred_dir.mkdir(exist_ok=True, parents=True)

            player_count_str_list = re.findall(r"\(players (\d+)\)", sol)
            player_count = (
                int(player_count_str_list[0]) if len(player_count_str_list) > 0 else 0
            )

            pred_program = extract_program(content)
            if pred_program == "":
                rewards.append(0.0)
                continue

            content_trial_dir = game_trial_dir / f"completion_{id}"
            content_concept_dir = game_concept_dir / f"completion_{id}"
            content_trial_dir.parent.mkdir(exist_ok=True, parents=True)
            content_concept_dir.parent.mkdir(exist_ok=True, parents=True)
            content_pred_path = game_pred_dir / f"completion_{id}.lud"
            with open(content_pred_path, "w") as f:
                f.write(pred_program)

            input_list.append(
                (
                    pred_program,
                    content_trial_dir,
                    content_concept_dir,
                    content_pred_path,
                    gamename,
                    player_count,
                )
            )

        process_num = len(input_list)
        if process_num == 0:
            return rewards
        with multiprocessing.Pool(process_num) as pool:
            concept_rewards = pool.map(process_concept_reward_func, input_list)

        rewards = rewards + concept_rewards

        return rewards

    return concept_reward


def process_grammar_reward(input, parser):
    content = input
    pred_program = extract_program(content)
    if pred_program == "":
        return 0.0

    try:
        parser.parse(pred_program)
        return 1.0
    except Exception as runtime_e:
        prefix, suffix = parser.handle_error(runtime_e)

    reward = len(prefix) / len(content)
    return reward


def get_grammar_reward(parser):
    def grammar_reward(completions, solution, **kwargs):
        rewards = []

        process_grammar_reward_func = partial(process_grammar_reward, parser=parser)
        input_list = []
        for completion, sol in zip(completions, solution):
            content = completion[0]["content"]
            pred_program = extract_program(content)
            if pred_program == "":
                rewards.append(0.0)
                continue

            input_list.append(content)

        process_num = len(input_list)
        if process_num == 0:
            return rewards
        with multiprocessing.Pool(process_num) as pool:
            grammar_rewards = pool.map(process_grammar_reward_func, input_list)

        rewards = rewards + grammar_rewards

        return rewards

    return grammar_reward


def get_repetition_penalty_reward(ngram_size: int, max_penalty: float):
    """
    Computes N-gram repetition penalty as described in Appendix C.2 of https://arxiv.org/abs/2502.03373.
    Reference implementation from: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

    Args:
    ngram_size: size of the n-grams
    max_penalty: Maximum (negative) penalty for wrong answers
    """
    if max_penalty > 0:
        raise ValueError(f"max_penalty {max_penalty} should not be positive")

    def zipngram(text: str, ngram_size: int):
        words = text.lower().split()
        return zip(*[words[i:] for i in range(ngram_size)])

    def repetition_penalty_reward(completions, **kwargs) -> float:
        """
        reward function the penalizes repetitions
        ref implementation: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

        Args:
            completions: List of model completions
        """

        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        for completion in contents:
            if completion == "":
                rewards.append(0.0)
                continue
            if len(completion.split()) < ngram_size:
                rewards.append(0.0)
                continue

            ngrams = set()
            total = 0
            for ng in zipngram(completion, ngram_size):
                ngrams.add(ng)
                total += 1

            scaling = 1 - len(ngrams) / total
            reward = scaling * max_penalty
            rewards.append(reward)
        return rewards

    return repetition_penalty_reward
