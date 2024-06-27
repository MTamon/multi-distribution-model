import json
import os
from argparse import ArgumentParser, Namespace


def gen_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--path", required=True, nargs="*", type=str)
    parser.add_argument("--out", required=True, type=str)

    args = parser.parse_args()

    return args


def process(args: Namespace):
    if os.path.isfile(args.out):
        os.remove(args.out)
    for input_path in args.path:
        with open(input_path, "r", encoding="utf-8") as f:
            total_duration = 0
            total_noise = 0
            for line in f:
                jdic = json.loads(line)

                duration = jdic["duration"]
                noise_section = jdic["overlap-section"]
                noise_duration = noise_section["end"] - noise_section["start"]

                total_duration += duration
                total_noise += noise_duration

        with open(args.out, "a", encoding="utf-8") as f:
            f.write(input_path + "\n")
            f.write(f"Noise duration rate : {(total_noise/ total_duration)}\n\n")


if __name__ == "__main__":
    _args = gen_args()
    process(_args)
