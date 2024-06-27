from typing import Callable

from nemo.collections.asr.models.rnnt_models import EncDecRNNTModel
from nemo.collections.asr.models.ctc_models import EncDecCTCModel
from nemo.collections.asr.data.audio_to_text_dali import DALIOutputs
from nemo.core.classes.mixins import AccessMixin

from omegaconf import DictConfig
from pytorch_lightning import Trainer

from torch.optim.lr_scheduler import _LRScheduler

from collections import OrderedDict

import torch
import torch.distributed

from nemo.collections.asr.parts.submodules.jasper import init_weights
from nemo.collections.asr.parts.utils import adapter_utils
from nemo.core.classes.common import typecheck
from nemo.core.classes.exportable import Exportable
from nemo.core.classes.mixins import adapter_mixins
from nemo.core.classes.module import NeuralModule
from nemo.core.neural_types import (
    AcousticEncodedRepresentation,
    LogprobsType,
    NeuralType,
)
from nemo.utils import logging


class MyEncDecRNNTModel(EncDecRNNTModel):
    """Extended EncDecRNNTModel"""

    def overwrite_scheduler(self, new_sched: _LRScheduler):
        """For overwriting _scheduler["scheduler"]."""
        self._scheduler["scheduler"] = new_sched


def get_overwrite_rnnt_sched_function() -> Callable[[_LRScheduler], None]:
    """Getting function which for overwrite `EncDecRNNTModel._scheduler`.\n
    use:
    `EncDecRNNTModel.overwrite_scheduler = get_overwrite_rnnt_sched_function()`

    Returns:
        Callable: Function which for overwrite `EncDecRNNTModel._scheduler
    """
    return MyEncDecRNNTModel.overwrite_scheduler


class FastEncDecRNNTModel(EncDecRNNTModel):
    """Skip computing train wer for faster training."""

    def __init__(self, cfg: DictConfig, trainer: Trainer = None, compute_wer: bool = False):
        super().__init__(cfg, trainer)
        self._optim_normalize_txu = None
        self.my_compute_wer = compute_wer

    # PTL-specific methods
    def training_step(self, batch, batch_nb):
        # Reset access registry
        if AccessMixin.is_access_enabled():
            AccessMixin.reset_registry(self)

        signal, signal_len, transcript, transcript_len = batch

        # forward() only performs encoder forward
        if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
            encoded, encoded_len = self.forward(processed_signal=signal, processed_signal_length=signal_len)
        else:
            encoded, encoded_len = self.forward(input_signal=signal, input_signal_length=signal_len)
        del signal

        # During training, loss must be computed, so decoder forward is necessary
        decoder, target_length, _ = self.decoder(targets=transcript, target_length=transcript_len)

        if hasattr(self, "_trainer") and self._trainer is not None:
            log_every_n_steps = self._trainer.log_every_n_steps
            sample_id = self._trainer.global_step
        else:
            log_every_n_steps = 1
            sample_id = batch_nb

        # If experimental fused Joint-Loss-WER is not used
        if not self.joint.fuse_loss_wer:
            # Compute full joint and loss
            joint = self.joint(encoder_outputs=encoded, decoder_outputs=decoder)
            loss_value = self.loss(
                log_probs=joint,
                targets=transcript,
                input_lengths=encoded_len,
                target_lengths=target_length,
            )

            # Add auxiliary losses, if registered
            loss_value = self.add_auxiliary_losses(loss_value)

            # Reset access registry
            if AccessMixin.is_access_enabled():
                AccessMixin.reset_registry(self)

            tensorboard_logs = {
                "train_loss": loss_value,
                "learning_rate": self._optimizer.param_groups[0]["lr"],
                "global_step": torch.tensor(self.trainer.global_step, dtype=torch.float32),
            }

            # Comment outed #############################################################
            # if self.compute_wer:
            #     if (sample_id + 1) % log_every_n_steps == 0:
            #         self.wer.update(encoded, encoded_len, transcript, transcript_len)
            #         _, scores, words = self.wer.compute()
            #         self.wer.reset()
            #         tensorboard_logs.update(
            #             {"training_batch_wer": scores.float() / words}
            #         )
            ##############################################################################

        else:
            # If experimental fused Joint-Loss-WER is used
            if (sample_id + 1) % log_every_n_steps == 0:
                # Changed ################################################################
                compute_wer = False  ###### True -> False #######
                ##########################################################################

            else:
                compute_wer = False

            # Fused joint step
            loss_value, wer, _, _ = self.joint(
                encoder_outputs=encoded,
                decoder_outputs=decoder,
                encoder_lengths=encoded_len,
                transcripts=transcript,
                transcript_lengths=transcript_len,
                compute_wer=compute_wer,
            )

            # Add auxiliary losses, if registered
            loss_value = self.add_auxiliary_losses(loss_value)

            # Reset access registry
            if AccessMixin.is_access_enabled():
                AccessMixin.reset_registry(self)

            tensorboard_logs = {
                "train_loss": loss_value,
                "learning_rate": self._optimizer.param_groups[0]["lr"],
                "global_step": torch.tensor(self.trainer.global_step, dtype=torch.float32),
            }

            if compute_wer:
                tensorboard_logs.update({"training_batch_wer": wer})

        # Log items
        self.log_dict(tensorboard_logs)

        # Preserve batch acoustic model T and language model U parameters if normalizing
        if self._optim_normalize_joint_txu:
            self._optim_normalize_txu = [encoded_len.max(), transcript_len.max()]

        return {"loss": loss_value}


class FastEncDecCTCModel(EncDecCTCModel):
    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        super().__init__(cfg, trainer)

    def training_step(self, batch, batch_nb):
        # Reset access registry
        if AccessMixin.is_access_enabled():
            AccessMixin.reset_registry(self)

        if self.is_interctc_enabled():
            AccessMixin.set_access_enabled(access_enabled=True)

        signal, signal_len, transcript, transcript_len = batch
        if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
            log_probs, encoded_len, _predictions = self.forward(
                processed_signal=signal, processed_signal_length=signal_len
            )
        else:
            log_probs, encoded_len, _predictions = self.forward(input_signal=signal, input_signal_length=signal_len)

        if hasattr(self, "_trainer") and self._trainer is not None:
            _log_every_n_steps = self._trainer.log_every_n_steps
        else:
            _log_every_n_steps = 1

        loss_value = self.loss(
            log_probs=log_probs,
            targets=transcript,
            input_lengths=encoded_len,
            target_lengths=transcript_len,
        )

        # Add auxiliary losses, if registered
        loss_value = self.add_auxiliary_losses(loss_value)
        # only computing WER when requested in the logs (same as done for final-layer WER below)
        loss_value, tensorboard_logs = self.add_interctc_losses(
            loss_value,
            transcript,
            transcript_len,
            # Changed ####################################################################
            # compute_wer=((batch_nb + 1) % log_every_n_steps == 0),
            compute_wer=False,
            ##############################################################################
        )

        # Reset access registry
        if AccessMixin.is_access_enabled():
            AccessMixin.reset_registry(self)

        tensorboard_logs.update(
            {
                "train_loss": loss_value,
                "learning_rate": self._optimizer.param_groups[0]["lr"],
                "global_step": torch.tensor(self.trainer.global_step, dtype=torch.float32),
            }
        )

        # Comment outed ##################################################################
        # if (batch_nb + 1) % log_every_n_steps == 0:
        #     self._wer.update(
        #         predictions=log_probs,
        #         targets=transcript,
        #         target_lengths=transcript_len,
        #         predictions_lengths=encoded_len,
        #     )
        #     wer, _, _ = self._wer.compute()
        #     self._wer.reset()
        #     tensorboard_logs.update({"training_batch_wer": wer})
        ##################################################################################

        return {"loss": loss_value, "log": tensorboard_logs}


class ConvASRDecoderMDIST(NeuralModule, Exportable, adapter_mixins.AdapterModuleMixin):
    """Simple ASR Decoder for use with CTC-based models such as JasperNet and QuartzNet

    Based on these papers:
       https://arxiv.org/pdf/1904.03288.pdf
       https://arxiv.org/pdf/1910.10261.pdf
       https://arxiv.org/pdf/2005.04290.pdf
    """

    @property
    def input_types(self):
        return OrderedDict({"encoder_output": NeuralType(("B", "D", "T"), AcousticEncodedRepresentation())})

    @property
    def output_types(self):
        return OrderedDict({"logprobs": NeuralType(("B", "T", "D"), LogprobsType())})

    def reduct_distribution(self, x: torch.Tensor):
        # x: [B, T, C]
        assert x.dim() == 3, f"Expected 3D tensor, got {x.dim()} ({x.shape})"
        batch_size, num_classes, seq_len = x.shape
        assert (
            num_classes == self._num_classes * self.distribution_num
        ), f"Expected {self._num_classes * self.distribution_num}, got {num_classes}"

        x = x.view(batch_size, self._num_classes, self.distribution_num, seq_len)
        x = torch.sum(x, dim=2)
        return x

    def __init__(self, feat_in, num_classes, init_mode="xavier_uniform", vocabulary=None, distribution_num=1):
        super().__init__()

        if vocabulary is None and num_classes < 0:
            raise ValueError(f"Neither of the vocabulary and num_classes are set! At least one of them need to be set.")

        if num_classes <= 0:
            num_classes = len(vocabulary)
            logging.info(f"num_classes of ConvASRDecoder is set to the size of the vocabulary: {num_classes}.")

        if vocabulary is not None:
            if num_classes != len(vocabulary):
                raise ValueError(
                    f"If vocabulary is specified, it's length should be equal to the num_classes. Instead got: num_classes={num_classes} and len(vocabulary)={len(vocabulary)}"
                )
            self.__vocabulary = vocabulary
        self._feat_in = feat_in
        # Add 1 for blank char
        self._num_classes = num_classes + 1

        # 変更箇所
        self.distribution_num = distribution_num
        self.output_size = self._num_classes * self.distribution_num
        # kernel_size=1 なので実質的にはLinear層
        self.decoder_layers = torch.nn.Sequential(
            torch.nn.Conv1d(self._feat_in, self.output_size, kernel_size=1, bias=True)
        )
        # 上手く行かなければ、softmax 後に reduct_distribution を適用してみる
        self.apply(lambda x: init_weights(x, mode=init_mode))

        accepted_adapters = [adapter_utils.LINEAR_ADAPTER_CLASSPATH]
        self.set_accepted_adapter_types(accepted_adapters)

        # to change, requires running ``model.temperature = T`` explicitly
        self.temperature = 1.0

    @typecheck()
    def forward(self, encoder_output):
        # Adapter module forward step
        if self.is_adapter_available():
            encoder_output = encoder_output.transpose(1, 2)  # [B, T, C]
            encoder_output = self.forward_enabled_adapters(encoder_output)
            encoder_output = encoder_output.transpose(1, 2)  # [B, C, T]

        if self.temperature != 1.0:
            return torch.nn.functional.log_softmax(
                self.reduct_distribution(self.decoder_layers(encoder_output)).transpose(1, 2) / self.temperature, dim=-1
            )
        return torch.nn.functional.log_softmax(
            self.reduct_distribution(self.decoder_layers(encoder_output)).transpose(1, 2), dim=-1
        )

    def input_example(self, max_batch=1, max_dim=256):
        """
        Generates input examples for tracing etc.
        Returns:
            A tuple of input examples.
        """
        input_example = torch.randn(max_batch, self._feat_in, max_dim).to(next(self.parameters()).device)
        return tuple([input_example])

    def _prepare_for_export(self, **kwargs):
        m_count = 0
        for m in self.modules():
            if type(m).__name__ == "MaskedConv1d":
                m.use_mask = False
                m_count += 1
        if m_count > 0:
            logging.warning(f"Turned off {m_count} masked convolutions")
        Exportable._prepare_for_export(self, **kwargs)

    # Adapter method overrides
    def add_adapter(self, name: str, cfg: DictConfig):
        # Update the config with correct input dim
        cfg = self._update_adapter_cfg_input_dim(cfg)
        # Add the adapter
        super().add_adapter(name=name, cfg=cfg)

    def _update_adapter_cfg_input_dim(self, cfg: DictConfig):
        cfg = adapter_utils.update_adapter_cfg_input_dim(self, cfg, module_dim=self._feat_in)
        return cfg

    @property
    def vocabulary(self):
        return self.__vocabulary

    @property
    def num_classes_with_blank(self):
        return self._num_classes
