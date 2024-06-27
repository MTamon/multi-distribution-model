import json
from argparse import ArgumentParser, Namespace


def gen_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--path", required=True, nargs="*", type=str)
    parser.add_argument("--keys", required=True, nargs="*", type=str)
    parser.add_argument("--output-path", required=True, type=str)

    args = parser.parse_args()

    return args


def process(args: Namespace):
    with open(args.output_path, "w", encoding="utf-8") as w:
        for input_file in args.path:
            with open(input_file, "r", encoding="utf-8") as f:
                for line in f:
                    jdic: dict = json.loads(line)

                    new_dic = {}
                    for key in args.keys:
                        new_dic[key] = jdic.get(key, None)

                    w.write(json.dumps(new_dic, indent=None, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    _args = gen_args()
    process(_args)
