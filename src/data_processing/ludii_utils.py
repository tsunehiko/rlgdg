import json
import re
import regex
from pathlib import Path
from dataclasses import dataclass
from textwrap import dedent


@dataclass
class Example:
    source: str
    target: str

    pred_grammar: str = None
    grammar: str = None
    label = None
    gamename: str = None
    gamepath: str = None
    train_examples = None


def list_to_xml(input_list: list, key: str):
    xml = ""
    if len(input_list) == 0:
        return xml
    if isinstance(input_list[0], dict):
        for id, item in enumerate(input_list):
            xml += f"\n<{key}_{id}>\n" + dict_to_xml(item) + f"\n</{key}_{id}>"
    elif isinstance(input_list[0], str):
        xml = ", ".join(input_list).strip()
    elif isinstance(input_list[0], list):
        for id, item in enumerate(input_list):
            xml += (
                f"\n<{key}_{id}>\n"
                + list_to_xml(item, f"{key}_{id}")
                + f"\n</{key}_{id}>"
            )
    return xml.strip()


def dict_to_xml(d: dict):
    xml = ""
    for k, v in d.items():
        if isinstance(v, dict):
            value_str = dict_to_xml(v)
        elif isinstance(v, list):
            value_str = list_to_xml(v, k)
        else:
            value_str = v
        xml += f"\n<{k}>\n{value_str.strip()}\n</{k}>"
    return xml.strip()


def load_ludii_meta(ludii_str):
    metadata_match = re.search(r"\(metadata\s*", ludii_str)
    if metadata_match is None:
        return {}
    metadata_str = ludii_str[metadata_match.end() :]

    metadata_pattern = r"\((?:[^()]|(?R))*\)"
    metadatas = regex.findall(metadata_pattern, metadata_str)
    if len(metadatas) == 0:
        return {}
    info_str = ""
    for metadata in metadatas:
        if metadata[1:5] == "info":
            info_str = metadata
            break
    info_str = info_str[5:-1].strip()

    info_pattern = r"\((?:[^()]|(?R))*\)"
    infos = regex.findall(info_pattern, info_str)
    metadata_dict = {}
    for info in infos:
        data = info.split('"')
        if len(data) < 2:
            continue

        key = data[0][1:].strip()
        value = '"'.join(data[1:-1])
        value = re.sub(r"\s+", " ", value)
        if key in metadata_dict:
            metadata_dict[key] += "\n" + value
        else:
            metadata_dict[key] = value

        if (
            "useFor_rules" not in metadata_dict
            and key == "useFor"
            and (
                "Reconstructed" in value
                or "Suggested" in value
                or "Described" in value
                or "Observed" in value
                or "Historical Information" in value
            )
        ):
            use_for_pattern = r"\((?:[^()]|(?R))*\)"
            use_for_list = regex.findall(use_for_pattern, value + ")")
            for use_for in use_for_list:
                if "rules" in use_for:
                    metadata_dict["useFor_rules"] = use_for[6:-1].strip()
                    break

    return metadata_dict


def load_ludii_from_file(filepath: Path):
    ludii, metadata = "", ""
    metadata_start = False
    with open(filepath, "r") as f:
        for line in f:
            if not metadata_start and line.startswith("//"):
                metadata_start = True
            elif metadata_start:
                metadata += line
            else:
                ludii += line
    metadata_dict = load_ludii_meta(metadata)
    return ludii, metadata_dict


def remove_newlines_in_parentheses(text):
    pattern = re.compile(r"\(\s*([^()\s]*?)\s*\n\s*([^()\s]*?)\s*\)", re.DOTALL)
    while True:
        new_text = pattern.sub(r"(\1\2)", text)
        if new_text == text:
            break
        text = new_text
    return new_text


def load_ludii_examples(filename, num_shot=-1):
    examples = []
    with open(filename, "r") as f:
        test_info = json.load(f)
    for gamepath, trainpaths in test_info.items():
        test_example = load_ludii_example(gamepath)
        train_examples = []
        if num_shot != -1:
            trainpaths = trainpaths[:num_shot]
        for trainpath in trainpaths:
            train_example = load_ludii_example(trainpath)
            train_examples.append(train_example)
        test_example.train_examples = train_examples
        examples.append(test_example)
    return examples


def load_ludii_example(gamepath):
    gamepath = gamepath.strip()
    gamename = gamepath.strip().split(".")[0].replace("/", "_")
    _, metadata = load_ludii_from_file(
        Path("data/ludii/Ludii/Common/res/lud") / gamepath
    )
    with open(Path("data/ludii/expand") / gamepath, "r") as f:
        expand_lud = f.read()
    expand_lud = remove_newlines_in_parentheses(expand_lud)
    grammar_path = Path("data/ludii/grammar") / gamepath.replace(".lud", ".txt")
    if grammar_path.exists():
        with open(grammar_path, "r") as f:
            grammar = f.read()
        grammar = grammar.strip()
    else:
        grammar = None
    source = {}
    if "description" in metadata:
        source["description"] = dedent(metadata["description"])
    if "rules" in metadata:
        source["rules"] = dedent(metadata["rules"])
    elif "useFor_rules" in metadata:
        source["rules"] = dedent(metadata["useFor_rules"])
    target = expand_lud.strip()
    example = Example(
        source=source,
        target=target,
        gamename=gamename,
        grammar=grammar,
        gamepath=gamepath,
    )
    return example
