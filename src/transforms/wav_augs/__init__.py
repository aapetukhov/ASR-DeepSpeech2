from src.transforms.wav_augs.gain import Gain
from src.transforms.wav_augs.noise import ColoredNoise
from src.transforms.wav_augs.pitch_shifting import PitchShifting
from src.transforms.wav_augs.speed_change import SpeedChange
from src.transforms.wav_augs.identity import Identity

all = [
    "Gain",
    "PitchShifting",
    "TimeStretching",
    "ColoredNoise",
    "Identity"
]