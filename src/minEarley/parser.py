import re
from typing import (
    List,
    Any,
    Collection,
)
from dataclasses import dataclass

import exrex
from lark.lexer import TerminalDef
from lark.load_grammar import load_grammar

from .earley import Parser
from .earley_exceptions import UnexpectedCharacters, UnexpectedEOF
from .utils import logger


@dataclass
class Option:
    start: List[str]
    keep_all_tokens: bool = True
    debug: bool = False
    oracle_rule: bool = False

    def __post_init__(self):
        if isinstance(self.start, str):
            self.start = [self.start]


@dataclass
class LexerConf:
    terminals: Collection[TerminalDef]
    ignore: Collection[str]
    use_bytes: bool = False
    re_module = re
    g_regex_flags: int = 0

    def __post_init__(self):
        self.terminals_by_name = {t.name: t for t in self.terminals}
        assert len(self.terminals) == len(self.terminals_by_name)


@dataclass
class ParserConf:
    rules: List[Any]
    start: List[str]


class EarleyParser:
    """
    Frontend for Earley parser
    """

    def __init__(self, grammar: str, **options) -> None:
        self.option = Option(**options)

        self.grammar, _ = load_grammar(
            grammar,
            source="<string>",
            import_paths=[],
            global_keep_all_tokens=self.option.keep_all_tokens,
        )
        self.terminals, self.rules, self.ignore_tokens = self.grammar.compile(
            self.option.start, terminals_to_keep=set()
        )

        self.lexer_conf = LexerConf(self.terminals, self.ignore_tokens)
        self.parser_conf = ParserConf(self.rules, self.option.start)
        self.parser = self.build_parser()

    def build_parser(self):
        parser = Parser(self.lexer_conf, self.parser_conf)
        return parser

    @classmethod
    def open(cls, grammar_filename: str, **options):
        with open(grammar_filename, encoding="utf-8") as f:
            return cls(f.read(), **options)

    def parse(self, text: str, start=None):
        if start is None:
            assert len(self.option.start) == 1, (
                "multiple start symbol, please specify one"
            )
            start = self.option.start[0]
        else:
            assert start in self.option.start

        return self.parser.parse(text, start)

    def handle_error(self, e):
        CANDIDATE_LIMIT = 128

        def regex_to_candidates(regex):
            candidates = set()
            if exrex.count(regex) > CANDIDATE_LIMIT:
                logger.warning(f"regex {regex} has too many candidates")
            for candidate in exrex.generate(regex, limit=CANDIDATE_LIMIT):
                candidates.add(candidate)
            return candidates

        def pattern_to_candidates(pattern):
            candidates = set()
            if pattern.type == "str":
                candidates.add(pattern.value)
            elif pattern.type == "regex":
                candidates.update(regex_to_candidates(pattern.value))
            elif pattern.type == "re" and self.option.oracle_rule:
                candidates.update(regex_to_candidates(pattern.value))
            return candidates

        if isinstance(e, UnexpectedCharacters):
            candidate_terminals = set()
            for terminal_name in e.allowed:
                terminal_def = self.lexer_conf.terminals_by_name[terminal_name]

                if (
                    terminal_name in ["INT_CONSTANT", "FLOAT_CONSTANT", "STRING"]
                    and terminal_def.pattern.type in ["regex", "re"]
                    and exrex.count(terminal_def.pattern.value) > CANDIDATE_LIMIT
                ):
                    new_candidates = set([terminal_name])
                else:
                    new_candidates = pattern_to_candidates(terminal_def.pattern)

                if (
                    terminal_def.pattern.type == "re"
                    and len(new_candidates) == 0
                    and not self.option.oracle_rule
                ):
                    new_candidates = set([terminal_def.name])

                candidate_terminals.update(new_candidates)
            prefix = e.parsed_prefix

            # TODO: handle case where no candidate is found
            if len(candidate_terminals) == 0:
                candidate_terminals = [""]
            return prefix, candidate_terminals
        elif isinstance(e, UnexpectedEOF):
            candidate_terminals = set()
            for terminal_name in e.expected:
                terminal_def = self.lexer_conf.terminals_by_name[terminal_name]

                if (
                    terminal_name in ["INT_CONSTANT", "FLOAT_CONSTANT", "STRING"]
                    and terminal_def.pattern.type in ["regex", "re"]
                    and exrex.count(terminal_def.pattern.value) > CANDIDATE_LIMIT
                ):
                    new_candidates = set([terminal_name])
                else:
                    new_candidates = pattern_to_candidates(terminal_def.pattern)

                if (
                    terminal_def.pattern.type == "re"
                    and len(new_candidates) == 0
                    and not self.option.oracle_rule
                ):
                    new_candidates = set([terminal_def.name])

                candidate_terminals.update(new_candidates)

            assert len(candidate_terminals) > 0
            prefix = e.text
            return prefix, candidate_terminals
        else:
            raise e
