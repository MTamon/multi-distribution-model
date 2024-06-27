# from nemo.core.optim.lr_scheduler import
from typing import Callable, Optional, Any
from typing_extensions import OrderedDict
from nemo.collections.asr.models.rnnt_models import EncDecRNNTModel
from nemo.collections.asr.models.ctc_models import EncDecCTCModel
from nemo.collections.asr.data.audio_to_text_dali import DALIOutputs
from nemo.core.classes.mixins import AccessMixin
import nemo.core.classes
from nemo.utils import logging
from nemo.utils.get_rank import get_rank
from omegaconf import DictConfig
import pytorch_lightning.loops.utilities
from pytorch_lightning import Trainer
from pytorch_lightning.loops.fit_loop import FitLoop, log
import pytorch_lightning.loops.optimization.optimizer_loop
from pytorch_lightning.loops.optimization.optimizer_loop import (
    OptimizerLoop,
    ClosureResult,
)
from pytorch_lightning.trainer.supporters import CombinedLoader
from pytorch_lightning.loops.utilities import _set_sampler_epoch
from pytorch_lightning.utilities.types import STEP_OUTPUT
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.memory import recursive_detach
import torch
from torch.optim.lr_scheduler import _LRScheduler


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

    def __init__(
        self, cfg: DictConfig, trainer: Trainer = None, compute_wer: bool = False
    ):
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
            encoded, encoded_len = self.forward(
                processed_signal=signal, processed_signal_length=signal_len
            )
        else:
            encoded, encoded_len = self.forward(
                input_signal=signal, input_signal_length=signal_len
            )
        del signal

        # During training, loss must be computed, so decoder forward is necessary
        decoder, target_length, _ = self.decoder(
            targets=transcript, target_length=transcript_len
        )

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
                "global_step": torch.tensor(
                    self.trainer.global_step, dtype=torch.float32
                ),
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
                "global_step": torch.tensor(
                    self.trainer.global_step, dtype=torch.float32
                ),
            }

            if compute_wer:
                tensorboard_logs.update({"training_batch_wer": wer})

        # Log items
        self.log_dict(tensorboard_logs)

        # Preserve batch acoustic model T and language model U parameters if normalizing
        if self._optim_normalize_joint_txu:
            self._optim_normalize_txu = [encoded_len.max(), transcript_len.max()]

        return {"loss": loss_value}


class SaveMemFitLoop(FitLoop):
    def __init__(
        self,
        min_epochs: Optional[int] = 0,
        max_epochs: Optional[int] = None,
    ) -> None:
        super().__init__(min_epochs, max_epochs)

    # fit_loop.py: 221
    def on_advance_start(self) -> None:
        """Prepares the dataloader for training and calls the hook ``on_train_epoch_start``"""
        model = self.trainer.lightning_module

        # reset train dataloader
        if (
            not self._is_fresh_start_epoch
            and self.trainer._data_connector._should_reload_train_dl
        ):
            log.detail(f"{self.__class__.__name__}: resetting train dataloader")
            self.trainer.reset_train_dataloader(model)
        self._is_fresh_start_epoch = False

        # reset outputs here instead of in `reset` as they are not accumulated between epochs
        del self._outputs
        self._outputs = []

        if self.trainer.train_dataloader is not None:
            assert isinstance(self.trainer.train_dataloader, CombinedLoader)
            _set_sampler_epoch(
                self.trainer.train_dataloader, self.epoch_progress.current.processed
            )

        # changing gradient according accumulation_scheduler
        self.trainer.accumulation_scheduler.on_train_epoch_start(
            self.trainer, self.trainer.lightning_module
        )

        # stores accumulated grad fractions per batch
        self.epoch_loop.batch_loop.accumulated_loss.reset(
            window_length=self.trainer.accumulate_grad_batches
        )

        self.epoch_progress.increment_ready()

        self.trainer._logger_connector.on_epoch_start()

        self.trainer._call_callback_hooks("on_train_epoch_start")
        self.trainer._call_lightning_module_hook("on_train_epoch_start")

        self.epoch_progress.increment_started()


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
            log_probs, encoded_len, predictions = self.forward(
                processed_signal=signal, processed_signal_length=signal_len
            )
        else:
            log_probs, encoded_len, predictions = self.forward(
                input_signal=signal, input_signal_length=signal_len
            )

        if hasattr(self, "_trainer") and self._trainer is not None:
            log_every_n_steps = self._trainer.log_every_n_steps
        else:
            log_every_n_steps = 1

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
            compute_wer=False
            ##############################################################################
        )

        # Reset access registry
        if AccessMixin.is_access_enabled():
            AccessMixin.reset_registry(self)

        tensorboard_logs.update(
            {
                "train_loss": loss_value,
                "learning_rate": self._optimizer.param_groups[0]["lr"],
                "global_step": torch.tensor(
                    self.trainer.global_step, dtype=torch.float32
                ),
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


class SaveMemOptimizerLoop(OptimizerLoop):
    def __init__(self) -> None:
        super().__init__()

    def _training_step(self, kwargs: OrderedDict) -> ClosureResult:
        """Performs the actual train step with the tied hooks.

        Args:
            kwargs: the kwargs passed down to the hooks.

        Returns:
            A ``ClosureResult`` containing the training step output.
        """
        # manually capture logged metrics
        training_step_output = self.trainer._call_strategy_hook(
            "training_step", *kwargs.values()
        )
        self.trainer.strategy.post_training_step()

        model_output = self.trainer._call_lightning_module_hook(
            "training_step_end", training_step_output
        )
        strategy_output = self.trainer._call_strategy_hook(
            "training_step_end", training_step_output
        )
        training_step_output = strategy_output if model_output is None else model_output

        del self._hiddens
        self._hiddens = _extract_hiddens(
            training_step_output, self.trainer.lightning_module.truncated_bptt_steps
        )

        result = self.output_result_cls.from_training_step_output(
            training_step_output, self.trainer.accumulate_grad_batches
        )

        if self.trainer.move_metrics_to_cpu:
            # hiddens and the training step output are not moved as they are not considered "metrics"
            assert self.trainer._results is not None
            self.trainer._results.cpu()

        return result


class SaveMemModelPT(nemo.core.classes.ModelPT):
    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        super().__init__(cfg, trainer)

    def on_train_batch_end(
        self, outputs, batch: Any, batch_idx: int, unused: int = 0
    ) -> None:
        """PyTorch Lightning hook:
        https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#on-train-batch-end
        We use it here to enable nsys profiling.
        """

        if self.device.type == "cuda":
            if hasattr(self, "_nsys_profile_enabled"):
                if self._nsys_profile_enabled:
                    if (
                        batch_idx == self._nsys_profile_end_step
                        and get_rank() in self._nsys_profile_ranks
                    ):
                        logging.info("====== End nsys profiling ======")
                        torch.cuda.cudart().cudaProfilerStop()
        del outputs
        del batch


def _extract_hiddens(
    training_step_output: STEP_OUTPUT, truncated_bptt_steps: int
) -> Optional[Any]:
    """Get the hidden state if present from the training step output.

    Raises:
        MisconfigurationException: If :attr:`~pytorch_lightning.core.Lightning.LightningModule.truncated_bptt_steps` is
            not enabled and hiddens are returned or vice versa.
    """
    if not truncated_bptt_steps:
        if isinstance(training_step_output, dict) and "hiddens" in training_step_output:
            raise MisconfigurationException(
                'You returned "hiddens" in your `training_step` but `truncated_bptt_steps` is disabled'
            )
        return None
    if (
        not isinstance(training_step_output, dict)
        or "hiddens" not in training_step_output
    ):
        raise MisconfigurationException(
            'You enabled `truncated_bptt_steps` but did not `return {..., "hiddens": ...}` in your `training_step`'
        )
    # detach hiddens to avoid `RuntimeError: Trying to backward through the graph a second time`
    hiddens = recursive_detach(training_step_output["hiddens"], to_cpu=True)
    return hiddens


def overwrite_save_gpu_memory():
    pytorch_lightning.loops.utilities._extract_hiddens = _extract_hiddens
    pytorch_lightning.loops.optimization.optimizer_loop.OptimizerLoop = (
        SaveMemOptimizerLoop
    )
    nemo.core.classes.ModelPT = SaveMemModelPT
