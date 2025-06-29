import re
import collections
from dataclasses import dataclass

from lark.load_grammar import _TERMINAL_NAMES
from minEarley.tree import Tree


def lark2bnf(grammar):
    grammar = grammar.replace(" : ", " ::= ")
    return grammar


inline_terminal_names = {
    # for SMC dataset
    "WORD",
    "NUMBER",
    "ESCAPED_STRING",
    "L",
    # for regex dataset
    "STRING",
    "INT",
    "CHARACTER_CLASS",
    "CONST",
    # for molecule
    "N",
    "C",
    "O",
    "F",
    "c",
    # for fol
    "PNAME",
    "CNAME",
    "LCASE_LETTER",
}
for k, v in _TERMINAL_NAMES.items():
    inline_terminal_names.add(v)


@dataclass
class SimpleRule:
    origin: str
    expansion: tuple

    def __hash__(self):
        return hash(str(self))

    def __str__(self):
        return self.to_lark()

    def to_lark(self):
        return f"{self.origin} : {' '.join(self.expansion)}"

    def to_bnf(self):
        return f"{self.origin} ::= {' '.join(self.expansion)}"

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, SimpleRule):
            return False
        return str(self).replace("?", "") == str(__o).replace("?", "")


def _wrap_string(s):
    if s.startswith('"') and s.endswith('"'):
        # a bit complex to preserve the quotation marks
        s = f'"\\{s[:-1]}\\"{s[-1]}'
    else:
        s = f'"{s}"'

    # escape unicode characters
    if "\\u" in s:
        s = s.replace("\\u", "\\\\u")

    return s


def split_rule(rule):
    split_idx = rule.index(":")
    lhs, rhs = rule[:split_idx].strip(), rule[split_idx + 1 :].strip()
    return lhs, rhs


def treenode2rule(treenode):
    if treenode is None:
        return None

    if isinstance(treenode, Tree):
        origin = f"{treenode.data.value}"
        expansion = []

        for child in treenode.children:
            if child is None:
                continue

            if isinstance(child, Tree):
                expansion.append(child.data.value)
            else:
                if child.type.startswith("__") or child.type in inline_terminal_names:
                    expansion.append(_wrap_string(child.value))
                else:
                    expansion.append(child.type)
    else:  # terminal
        if treenode.type.startswith("__") or treenode.type in inline_terminal_names:
            return None
        else:
            origin = treenode.type
            expansion = [_wrap_string(treenode.value)]
    return SimpleRule(origin, tuple(expansion))


def tree2rulelist(tree):
    rule_list = []

    def recur_add(node, rule_list):
        cur_rule = treenode2rule(node)
        if cur_rule is None:
            return
        rule_list.append(cur_rule)

        if getattr(node, "children", None):
            for child in node.children:
                recur_add(child, rule_list)

    recur_add(tree, rule_list)
    return rule_list


def extract_rule_stat(tree, rule_stat):
    """
    Count the occurrence of each rule
    """
    cur_rule = treenode2rule(tree)
    if cur_rule is None:
        return
    if cur_rule not in rule_stat:
        rule_stat[cur_rule] = 1
    else:
        rule_stat[cur_rule] += 1

    if getattr(tree, "children", None):
        for child in tree.children:
            extract_rule_stat(child, rule_stat)


def rulelist2larkstr(rule_stat):
    lhs2rhs = collections.OrderedDict()
    for rule in rule_stat:
        lhs, rhs = rule.origin, rule.expansion
        if lhs not in lhs2rhs:
            lhs2rhs[lhs] = []
        lhs2rhs[lhs].append(rhs)

    grammar = ""
    for lhs in lhs2rhs:
        grammar += f"{lhs} :"
        for rhs in lhs2rhs[lhs]:
            rhs_str = " ".join(rhs)
            grammar += f" {rhs_str} |"
        grammar = grammar[:-2]
        grammar += "\n"

    return grammar.strip()


def extract_min_grammar_from_trees(trees, return_rules=False):
    """
    Extract minimal grammar to reconstruct the tree
    """
    rule_stat = collections.OrderedDict()
    for tree in trees:
        extract_rule_stat(tree, rule_stat)
    grammar = rulelist2larkstr(rule_stat)

    if return_rules:
        return grammar, list(rule_stat.keys())
    else:
        return grammar


def remove_newlines_in_parentheses(text):
    pattern = re.compile(r"\(\s*([^()\s]*?)\s*\n\s*([^()\s]*?)\s*\)", re.DOTALL)
    while True:
        new_text = pattern.sub(r"(\1\2)", text)
        if new_text == text:
            break
        text = new_text
    return new_text


def gen_ludii_min_lark(program, parser):
    parse_trees = []
    program = remove_newlines_in_parentheses(program)
    parse_tree = parser.parse(program)
    parse_trees.append(parse_tree)
    grammar = extract_min_grammar_from_trees(parse_trees)
    return grammar
