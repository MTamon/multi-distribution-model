import os
import datetime
import pytorch_lightning as pl
from pytorch_lightning import Callback
from nemo.core.classes import ModelPT


class SavedCallback(Callback):
    def __init__(
        self,
        save_dir="models",
        saved_name: str = "model.nemo",
        monitor: str = "val_wer",
        top_k: int = -1,
    ) -> None:
        super().__init__()
        self._saved_name = saved_name
        self._top_k = top_k

        self._monitor = monitor
        self._prev_monitor = {}

        # self.save_dir = os.path.join(
        #     save_dir, datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        # )
        self.save_dir = save_dir
        self.logfile = os.path.join(self.save_dir, "log.log")

        if not os.path.isdir(self.save_dir):
            os.mkdir(self.save_dir)

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "ModelPT") -> None:
        if self._top_k == -1:
            return

        _save_name, _ext = self._saved_name.rsplit(".", maxsplit=1)
        _save_name = _save_name + "E" + str(trainer.current_epoch) + "." + _ext
        _save_name = os.path.join(self.save_dir, _save_name)

        if not os.path.isfile(_save_name):
            with open(self.logfile, "a", encoding="utf-8") as f:
                f.write(datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S "))
                f.write(f"save model ... {os.path.basename(_save_name)}\n")
            pl_module.save_to(_save_name)

    def on_validation_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "ModelPT"
    ) -> None:
        metrics = trainer.callback_metrics[self._monitor]

        _save_name, _ext = self._saved_name.rsplit(".", maxsplit=1)
        _save_name = _save_name + "L" + str(float(metrics)) + "." + _ext
        _save_name = os.path.join(self.save_dir, _save_name)

        if len(self._prev_monitor) < self._top_k or self._top_k == -1:
            self._prev_monitor[_save_name] = metrics

            if not os.path.isfile(_save_name):
                with open(self.logfile, "a", encoding="utf-8") as f:
                    f.write(datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S "))
                    f.write(f"save model ... {os.path.basename(_save_name)}\n")
                pl_module.save_to(_save_name)
        else:
            sorted_prev = sorted(self._prev_monitor.items(), key=lambda x: x[1])
            lowest = sorted_prev[-1][0]

            lowest_metrics = self._prev_monitor.pop(lowest)

            if lowest_metrics > metrics:
                self._prev_monitor[_save_name] = metrics

                if os.path.isfile(_save_name):
                    with open(self.logfile, "a", encoding="utf-8") as f:
                        f.write(datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S "))
                        f.write(f"remove model ... {os.path.basename(lowest)}\n")
                    os.remove(lowest)
                if not os.path.isfile(_save_name):
                    with open(self.logfile, "a", encoding="utf-8") as f:
                        f.write(datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S "))
                        f.write(f"save model ... {os.path.basename(_save_name)}\n")
                    pl_module.save_to(_save_name)
            else:
                self._prev_monitor[lowest] = lowest_metrics
