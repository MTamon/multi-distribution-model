# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
# Training the model

Basic run (on CPU for 50 epochs):
    python examples/asr/asr_ctc/speech_to_text_ctc.py \
        # (Optional: --config-path=<path to dir of configs> --config-name=<name of config without .yaml>) \
        model.train_ds.manifest_filepath="<path to manifest file>" \
        model.validation_ds.manifest_filepath="<path to manifest file>" \
        trainer.devices=1 \
        trainer.accelerator='cpu' \
        trainer.max_epochs=50


Add PyTorch Lightning Trainer arguments from CLI:
    python speech_to_text_ctc.py \
        ... \
        +trainer.fast_dev_run=true

Hydra logs will be found in "$(./outputs/$(date +"%y-%m-%d")/$(date +"%H-%M-%S")/.hydra)"
PTL logs will be found in "$(./outputs/$(date +"%y-%m-%d")/$(date +"%H-%M-%S")/lightning_logs)"

Override some args of optimizer:
    python speech_to_text_ctc.py \
    # (Optional: --config-path=<path to dir of configs> --config-name=<name of config without .yaml>) \
    model.train_ds.manifest_filepath="./an4/train_manifest.json" \
    model.validation_ds.manifest_filepath="./an4/test_manifest.json" \
    trainer.devices=2 \
    trainer.max_epochs=2 \
    model.optim.args.betas=[0.8,0.5] \
    model.optim.args.weight_decay=0.0001

Override optimizer entirely
    python speech_to_text_ctc.py \
    # (Optional: --config-path=<path to dir of configs> --config-name=<name of config without .yaml>) \
    model.train_ds.manifest_filepath="./an4/train_manifest.json" \
    model.validation_ds.manifest_filepath="./an4/test_manifest.json" \
    trainer.devices=2 \
    trainer.max_epochs=2 \
    model.optim.name=adamw \
    model.optim.lr=0.001 \
    ~model.optim.args \
    +model.optim.args.betas=[0.8,0.5]\
    +model.optim.args.weight_decay=0.0005

# Fine-tune a model

For documentation on fine-tuning this model, please visit -
https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/configs.html#fine-tuning-configurations

# Pretrained Models

For documentation on existing pretrained models, please visit -
https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/results.html

"""
import os

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf, open_dict

# from nemo.collections.asr.models import EncDecCTCModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager

from utils.overwrite_tools import FastEncDecCTCModel, ConvASRDecoderMDIST
from utils.manifest import get_ds_tokens
import nemo.collections.asr.modules


def load_vocab(vocab_path):
    vocab = []
    with open(vocab_path, "r", encoding="utf-8") as f:
        for token in f:
            token = token.rsplit("\n", maxsplit=1)[0]
            vocab.append(token)
    return vocab


@hydra_runner(config_path="./", config_name="conformer_ctc_char.yaml")
def main(cfg):
    logging.info(f"Hydra config: {OmegaConf.to_yaml(cfg)}")

    # Override the ASR decoder to use the multi-distribution decoder
    nemo.collections.asr.modules.ConvASRDecoder = ConvASRDecoderMDIST

    if cfg.vocab is None:
        tokens = sorted(list(get_ds_tokens(cfg)))
        with open(f"{cfg.exp_dir}/vocab_CTC.txt", "w", encoding="utf-8") as f:
            for token in tokens:
                f.write(token + "\n")
    else:
        tokens = sorted(load_vocab(cfg.vocab))

    with open_dict(cfg):
        cfg.model.labels = tokens

    if not os.path.exists(cfg.exp_dir):
        os.mkdir(cfg.exp_dir)

    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))
    asr_model = FastEncDecCTCModel(cfg=cfg.model, trainer=trainer)

    if cfg.mode == "Finetune":
        if cfg.nemo_path.rsplit(".", 1)[-1] == "nemo":
            model_ckpt = FastEncDecCTCModel.extract_state_dict_from(cfg.nemo_path, cfg.exp_dir)
            asr_model.load_state_dict(model_ckpt)
            del model_ckpt
        elif cfg.nemo_path.rsplit(".", 1)[-1] == "ckpt":
            state_dict = torch.load(cfg.nemo_path)["state_dict"]
            asr_model.load_state_dict(state_dict=state_dict)
            asr_model.change_vocabulary(tokens)

    # Initialize the weights of the model from another model, if provided via config
    asr_model.maybe_init_from_pretrained_checkpoint(cfg)

    yaml_txt = OmegaConf.to_yaml(cfg)
    with open(f"{cfg.exp_dir}/confCTC.yaml", "w", encoding="utf-8") as _f:
        _f.write(yaml_txt)
    print(asr_model.summarize())

    trainer.fit(asr_model)

    if hasattr(cfg.model, "test_ds") and cfg.model.test_ds.manifest_filepath is not None:
        if asr_model.prepare_test(trainer):
            trainer.test(asr_model)


if __name__ == "__main__":
    main()  # noqa pylint: disable=no-value-for-parameter
