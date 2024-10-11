from src.transforms.wav_augs.gain import Gain
from src.transforms.wav_augs.noise import ColoredNoise
from src.transforms.wav_augs.pitch_shifting import PitchShifting
from src.transforms.wav_augs.time_stretching import TimeStretching
from src.transforms.wav_augs.identity import Identity

all = [
    "Gain",
    "PitchShifting",
    "TimeStretching",
    "ColoredNoise",
    "Identity"
]