"""For overwrite _AudioTextDataset __getitem__"""

from argparse import Namespace
from nemo.collections.asr.data.audio_to_text import _AudioTextDataset
import torch
import numpy as np

np.random.seed(0)


def set_overwrite(args: Namespace):
    """Setting _AudioTextDataset overwrite"""
    _AudioTextDataset.__getitem__ = _OverlapAudioTextDataset.__getitem__
    _AudioTextDataset.snr = float(args.snr)


class _OverlapAudioTextDataset(_AudioTextDataset):
    def __getitem__(self, index):
        if not "snr" in dir(self):
            raise AttributeError(
                "You have to run set_overwrite() before make pytorch_lightning.Trainer"
            )

        ovl_index = torch.randint(
            low=0, high=len(self.manifest_processor.collection), size=(1,)
        )

        sample = self.manifest_processor.collection[index]
        overlap = self.manifest_processor.collection[ovl_index]
        offset = sample.offset
        ovl_offset = overlap.offset

        if offset is None:
            offset = 0
        if ovl_offset is None:
            ovl_offset = 0

        features = self.featurizer.process(
            sample.audio_file,
            offset=offset,
            duration=sample.duration,
            trim=self.trim,
            orig_sr=sample.orig_sr,
            channel_selector=self.channel_selector,
        )
        f, fl = features, torch.tensor(features.shape[0]).long()
        ovl_features = self.featurizer.process(
            overlap.audio_file,
            offset=ovl_offset,
            duration=overlap.duration,
            trim=self.trim,
            orig_sr=overlap.orig_sr,
            channel_selector=self.channel_selector,
        )
        ovl_f, ovl_fl = ovl_features, torch.tensor(ovl_features.shape[0]).long()

        f = overlap_voice(self.snr, f, fl, ovl_f, ovl_fl)

        t, tl = self.manifest_processor.process_text_by_sample(sample=sample)

        if self.return_sample_id:
            output = f, fl, torch.tensor(t).long(), torch.tensor(tl).long(), index
        else:
            output = f, fl, torch.tensor(t).long(), torch.tensor(tl).long()

        return output


def overlap_voice(
    snr: float,
    f: torch.Tensor,
    fl: torch.Tensor,
    ovl_f: torch.Tensor,
    ovl_fl: torch.Tensor,
):
    use_dtype = f.dtype

    if fl > ovl_fl:
        forward_len = torch.randint(low=0, high=(fl - ovl_fl + 1), size=(1,))
        backward_len = fl - ovl_fl - forward_len
        forward = torch.zeros(size=(forward_len,), dtype=use_dtype)
        backward = torch.zeros(size=(backward_len,), dtype=use_dtype)
        ovl_f = torch.cat([forward, ovl_f, backward])
    if fl < ovl_fl:
        idx = torch.randint(low=0, high=(ovl_fl - fl), size=(1,))
        ovl_f = ovl_f[idx:fl]

    rms = (f**2).mean(dtype=torch.float32).sqrt()
    ovl_rms = (ovl_f**2).mean(dtype=torch.float32).sqrt()

    a = snr / 20
    target_ovl_rms = rms / (10**a)
    adj_ovl_f = ovl_f * (target_ovl_rms / ovl_rms)

    mix_f = f + adj_ovl_f

    if mix_f.max(axis=0) > 32767:
        mix_f = mix_f * (32767 / mix_f.max(axis=0))
        f = f * (32767 / mix_f.max(axis=0))
        adj_ovl_f = adj_ovl_f * (32767 / mix_f.max(axis=0))

    return mix_f
