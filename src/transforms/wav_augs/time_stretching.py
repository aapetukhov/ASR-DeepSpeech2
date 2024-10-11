import torchaudio
from torch import Tensor, nn


class TimeStretching(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._aug = torchaudio.transforms.TimeStretch(*args, **kwargs)

    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        return self._aug(x).squeeze(1) 