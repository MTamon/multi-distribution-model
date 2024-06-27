"""Conformer-Transducer (ASJ edit.)"""
import wandb
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf, open_dict, DictConfig
from pytorch_lightning.loggers.wandb import WandbLogger

from callback import SavedCallback
from utils.data_module import set_overwrite
from utils.argments import get_asj_args
from utils.manifest import get_tokens
from utils.overwrite_tools import FastEncDecCTCModel

# Save GPU Memory
# overwrite_save_gpu_memory()

# torch.backends.cuda.cufft_plan_cache[0].max_size = 10
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

args = get_asj_args()
if args.overlap_on_time:
    set_overwrite(args)

CONFIG_PATH = args.nemo_path
CKPT_NAME = args.ckpt_name
CKPT_DIR = args.ckpt_dir
TRAIN_MANIFEST = args.train_path
TEST_MANIFEST = args.test_path
VALID_MANIFEST = args.valid_path

PROJ_NAME = args.proj_name

EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
ACCUMULATION = args.ga

DATA_NUM = len(open(TRAIN_MANIFEST, "r", encoding="utf-8").readlines())
ITERATION_NUM = DATA_NUM / BATCH_SIZE
EPOCH_STEP = int(DATA_NUM / (BATCH_SIZE * ACCUMULATION))

LEARNING_RATE = args.lr
MIN_LR = args.min_lr
WEIGHT_DECAY = args.weight_decay
ADAMW_BETA = (0.9, 0.98)

SCHEDULER = args.scheduler
D_MODEL = args.d_model
CONSTANT_STEPS = EPOCH_STEP * args.constant_epochs
WARMUP_EPOCHS = args.warmup_epochs
# MAX_STEP = (EPOCH_STEP * args.max_epoch) if args.max_epoch is not None else None
MAX_STEP = None
WARMUP_RATE = None
# WARMUP_STEP = EPOCH_STEP * WARMUP_EPOCHS
WARMUP_STEP = args.warmup_steps
DEVICES = args.devices
ACCELERATOR = args.accelerator
STRATEGY = args.strategy


def set_conf(config):
    with open_dict(config.cfg):
        config.cfg.train_ds.manifest_filepath = TRAIN_MANIFEST
        config.cfg.train_ds.batch_size = BATCH_SIZE
        config.cfg.train_ds.num_workers = BATCH_SIZE

        config.cfg.test_ds.manifest_filepath = TEST_MANIFEST
        config.cfg.test_ds.batch_size = BATCH_SIZE
        config.cfg.test_ds.num_workers = BATCH_SIZE

        config.cfg.validation_ds.manifest_filepath = VALID_MANIFEST
        config.cfg.validation_ds.batch_size = BATCH_SIZE
        config.cfg.validation_ds.num_workers = BATCH_SIZE
        config.cfg.validation_ds.pin_memory = True
        config.cfg.validation_ds.trim_silence = True

        config.cfg.joint.fused_batch_size = BATCH_SIZE

        config.cfg.optim.lr = LEARNING_RATE
        config.cfg.optim.betas = ADAMW_BETA  # from paper
        config.cfg.optim.weight_decay = WEIGHT_DECAY  # Original weight decay
        config.cfg.optim.sched.name = SCHEDULER
        config.cfg.optim.sched.constant_steps = CONSTANT_STEPS
        config.cfg.optim.sched.min_lr = MIN_LR
        config.cfg.optim.sched.warmup_steps = WARMUP_STEP
        config.cfg.optim.sched.warmup_ratio = WARMUP_RATE  # 5 % warmup
        config.cfg.optim.sched.max_steps = MAX_STEP

        if SCHEDULER == "CosineAnnealing":
            if config.cfg.optim.sched.get("d_model", None) is not None:
                config.cfg.optim.sched.pop("d_model")
        if SCHEDULER == "NoamAnnealing":
            if config.cfg.optim.sched.get("constant_steps", None) is not None:
                config.cfg.optim.sched.pop("constant_steps")
            if config.cfg.optim.sched.get("d_model", None) is None:
                config.cfg.optim.sched.d_model = D_MODEL
    return config


def load_vocab(vocab_path):
    vocab = []
    with open(vocab_path, "r", encoding="utf-8") as f:
        for token in f:
            token = token.rsplit("\n", maxsplit=1)[0]
            vocab.append(token)
    return vocab


callback = SavedCallback(
    CKPT_DIR,
    CKPT_NAME,
    monitor="val_wer",
    top_k=args.top_k,
)


# initialise the wandb logger and name your wandb project
wandb_logger = WandbLogger(project=PROJ_NAME)  #####
# add your batch size to the wandb config
# wandb_logger.experiment.config["batch_size"] = BATCH_SIZE

torch.manual_seed(0)

trainer = pl.Trainer(
    accumulate_grad_batches=ACCUMULATION,
    devices=DEVICES,
    accelerator=ACCELERATOR,
    strategy=STRATEGY,
    num_nodes=1,
    max_epochs=500,
    max_steps=-1,
    benchmark=False,  # needs to be false for models with variable-length speech input as it slows down training
    num_sanity_val_steps=0,
    logger=wandb_logger,
    enable_checkpointing=True,
    callbacks=callback,
    log_every_n_steps=int(EPOCH_STEP / args.epoch_logging_n),
    precision=args.precision,
    sync_batchnorm=True,
    # val_check_interval=1,
    check_val_every_n_epoch=1,
    # amp_backend="apex",
    # amp_level="O1",
)

model = OmegaConf.load(args.yaml_path)
set_conf(model)
conf_ctc = FastEncDecCTCModel(cfg=model.cfg)
if args.vocab is None:
    conf_ctc.change_vocabulary(new_vocabulary=list(get_tokens(args)))
else:
    conf_ctc.change_vocabulary(new_vocabulary=load_vocab(args.vocab))

if CONFIG_PATH is not None:
    model_ckpt = FastEncDecCTCModel.extract_state_dict_from(CONFIG_PATH, args.exp_dir)
    conf_ctc.load_state_dict(model_ckpt)

exp_cfg = conf_ctc.cfg.copy()
exp_cfg = DictConfig({"cfg": exp_cfg})
yaml_txt = OmegaConf.to_yaml(exp_cfg)
with open("./asj_exp/pretrain/confT.yaml", "w", encoding="utf-8") as _f:
    _f.write(yaml_txt)

# conf_ctc._wer.use_cer = True
# conf_ctc._wer.log_prediction = True
# conf_ctc.compute_eval_loss = True

conf_ctc.set_trainer(trainer)
conf_ctc.setup_training_data(conf_ctc.cfg.train_ds)
conf_ctc.setup_multiple_test_data(conf_ctc.cfg.test_ds)
conf_ctc.setup_multiple_validation_data(conf_ctc.cfg.validation_ds)
optimizer, _ = conf_ctc.setup_optimization(conf_ctc.cfg.optim)

print(conf_ctc.summarize())

trainer.fit(conf_ctc)


wandb.finish()
