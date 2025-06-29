from llms.models.gemma import Gemma
from llms.models.gpt import ChatGPT
from llms.models.hf_model import HFModel
from llms.models.llm import LLMConfig


def setup_llm(args: LLMConfig):
    split_point = args.engine.index("/")
    args.platform, args.model_name = (
        args.engine[:split_point],
        args.engine[split_point + 1 :],
    )
    if args.platform == "openai":
        llm = ChatGPT(args)
    elif args.platform in ["local", "hf"]:
        if args.platform == "local":
            group, _ = args.tokenizer.split("/")
        else:
            group, _ = args.model_name.split("/")

        if group == "google":
            llm = Gemma(args)
        else:
            llm = HFModel(args)
    else:
        raise NotImplementedError(f"platform {args.platform} not supported")
    return llm
