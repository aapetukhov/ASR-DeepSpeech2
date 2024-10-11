import torch.nn as nn
from torch import Tensor
import random

from wav_augs import Identity, Gain, PitchShifting, TimeStretching, ColoredNoise
from hydra.utils import instantiate


class RandomTransform(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.transforms = []
        self.probs = []
        for transform_cfg in args:
            transform = instantiate(transform_cfg)
            prob = transform_cfg.get("p", 1.0)

            self.transforms.append(transform)
            self.probs.append(prob)

        total_prob = sum(self.probs)
        assert abs(total_prob - 1.0) < 1e-6, f"Probs have to add up to 1, yours add up to {total_prob}"

    def __call__(self, data: Tensor) -> Tensor:
        transform = random.choices(self.transforms, self.probs)[0]
        return transform(data)