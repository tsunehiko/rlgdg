from minEarley.parser import EarleyParser
from minEarley.earley_exceptions import UnexpectedInput


def parse_with_lark(sentence):
    from lark import Lark

    parser = Lark.open(
        "grammars/cfg.lark",
        start="sentence",
        ambiguity="explicit",
        keep_all_tokens=True,
    )
    t = parser.parse(sentence)
    print(t.pretty())


if __name__ == "__main__":
    sentence = "fruit flies like a banana"
    parse_with_lark(sentence)

    parser = EarleyParser.open("grammars/cfg.lark", start="sentence")
    try:
        parse = parser.parse(sentence)
        print(parse.pretty())
    except UnexpectedInput as parse_error:
        candidates = parser.handle_error(parse_error)
        print("Parsing failed, but here are the candidates for correction:")
        print(candidates)
