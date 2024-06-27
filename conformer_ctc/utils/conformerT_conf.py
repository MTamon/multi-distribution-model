"""conformerT config setup"""
from argparse import Namespace
from manifest import build_charset


def get_tokens(args: Namespace):
    charset = build_charset(**vars(args))

    return set(charset["train"].keys())
