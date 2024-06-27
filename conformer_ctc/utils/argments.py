from argparse import ArgumentParser, Namespace


def add_cejc_args(parser: ArgumentParser):
    parser.add_argument(
        "--manifest-path",
        default=None,
        type=str,
        help="Path of CEJC manifest files",
    )
    parser.add_argument(
        "--new-manifest-path",
        default=None,
        type=str,
        help="Output path of new CEJC manifest files",
    )

    return parser


def get_cejc_args() -> Namespace:
    """get argments for making dataset from CEJC.

    Returns:
        Namespace
    """
    parser = ArgumentParser("This program to remove utt-tag from CEJC.")
    parser = add_cejc_args(parser)

    return parser.parse_args()


def add_merge_args(parser: ArgumentParser):
    parser.add_argument(
        "--cejc-path",
        default=None,
        type=str,
        help="Path of CEJC manifest files",
    )
    parser.add_argument(
        "--csj-path",
        default=None,
        type=str,
        help="Path of CSJ manifest files",
    )
    parser.add_argument(
        "--merged-path",
        default=None,
        type=str,
        help="Output path of new Merged ( CEJC + CSJ ) manifest files",
    )

    return parser


def get_merge_args() -> Namespace:
    """get argments for making dataset from CSJ + CEJC.

    Returns:
        Namespace
    """
    parser = ArgumentParser("This program is for making dataset from CSJ + CEJC.")
    parser = add_merge_args(parser)

    return parser.parse_args()


def add_finetune_args(parser: ArgumentParser):
    parser.add_argument(
        "--proj-name",
        default="No titled",
        type=str,
        help="Project Name",
    )
    parser.add_argument(
        "--trial-name",
        default="trial",
        type=str,
        help="WandB trial Name",
    )
    parser.add_argument(
        "--nemo-path",
        default=None,
        type=str,
        help="Path of .nemo config file",
    )
    parser.add_argument(
        "--ckpt-name",
        default=None,
        type=str,
        help="Path of .nemo config file",
    )
    parser.add_argument(
        "--ckpt-dir",
        default=None,
        type=str,
        help="Path of .nemo config file",
    )
    parser.add_argument(
        "--train-path",
        default=None,
        type=str,
        help="Path of train manifest files",
    )
    parser.add_argument(
        "--test-path",
        default=None,
        type=str,
        help="Path of test manifest files",
    )
    parser.add_argument(
        "--valid-path",
        default=None,
        type=str,
        help="Path of validation manifest files",
    )
    parser.add_argument(
        "--overlap-on-time",
        default=False,
        action="store_true",
        help="Run overlap on time.",
    )
    parser.add_argument(
        "--snr",
        default=5.0,
        type=float,
        help="Signal to Noise Rate",
    )
    parser.add_argument(
        "--scheduler",
        default="NoamAnnealing",
        type=str,
        help="Available: 'CosineAnnealing', 'NoamAnnealing'",
    )
    parser.add_argument(
        "--lr",
        default=0.01,
        type=float,
        help="Learning rate",
    )
    parser.add_argument(
        "--min-lr",
        default=1e-5,
        type=float,
        help="Min-learning rate",
    )
    parser.add_argument(
        "--d-model",
        default=512,
        type=int,
        help="Learning rate scheduling parameter.",
    )
    parser.add_argument(
        "--max-epoch",
        default=None,
        type=int,
        help="Learning rate scheduling parameter.",
    )
    parser.add_argument(
        "--constant-epochs",
        default=1,
        type=int,
        help="Learning rate scheduling parameter.",
    )
    parser.add_argument(
        "--warmup-epochs",
        default=None,
        type=int,
        help="Learning rate scheduling parameter.",
    )
    parser.add_argument(
        "--warmup-steps",
        default=None,
        type=int,
        help="Learning rate scheduling parameter.",
    )
    parser.add_argument(
        "--weight-decay",
        default=0.001,
        type=float,
        help="Weight decay. Default to 0.001.",
    )
    parser.add_argument(
        "--ga",
        default=1,
        type=int,
        help="Grad accumlation.",
    )
    parser.add_argument(
        "--batch-size",
        default=1,
        type=int,
        help="Batch size.",
    )
    parser.add_argument(
        "--epochs",
        default=10,
        type=int,
        help="Epoch num.",
    )
    parser.add_argument(
        "--top-k",
        default=5,
        type=int,
        help="Saved top-k models.",
    )
    parser.add_argument(
        "--epoch-logging-n",
        default=5,
        type=int,
        help="Log outputting times per epoch.",
    )
    parser.add_argument(
        "--devices",
        default=1,
        type=int,
        help="GPU num.",
    )
    parser.add_argument(
        "--accelerator",
        default="gpu",
        type=str,
        help="GPU types.",
    )
    parser.add_argument(
        "--strategy",
        default=None,
        type=str,
        help="Processor strategy.",
    )

    return parser


def get_finetune_args() -> Namespace:
    """get argments for finetuning.

    Returns:
        Namespace
    """
    parser = ArgumentParser("This program is for finetuning.")
    parser = add_finetune_args(parser)

    return parser.parse_args()


def add_confT_args(parser: ArgumentParser):
    parser.add_argument(
        "--data-path",
        default=None,
        type=str,
        help="Path of target manifest files",
    )
    parser.add_argument(
        "--yaml-path",
        default=None,
        type=str,
        help="Path of yaml config file",
    )
    parser.add_argument(
        "--train-dir-name",
        default=None,
        type=str,
        help="Name of train dataset directory",
    )
    parser.add_argument(
        "--valid-dir-name",
        default=None,
        type=str,
        help="Name of valid dataset directory",
    )
    parser.add_argument(
        "--exp-dir",
        default="./asj_exp/finetune",
        type=str,
        help="Path to save dir for Fine-tuning .ckpt",
    )
    parser.add_argument(
        "--vocab",
        default=None,
        type=str,
        help="Path to save dir for vocab",
    )
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        help="Use precision",
    )
    parser.add_argument(
        "--compute-wer",
        default=False,
        action="store_true",
        help="Compute wer for every logs.",
    )

    return parser


def get_asj_args() -> Namespace:
    parser = ArgumentParser("For ASJ.")
    parser = add_finetune_args(parser)
    parser = add_confT_args(parser)

    return parser.parse_args()
