#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2023 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""contains augmentations"""
import os
import random
import sentencepiece
import torch
import torchaudio
import torchvision


NOISE_FILENAME = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "babble_noise.wav"
)

SP_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "spm",
    "unigram",
    "unigram5000.model",
)

DICT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "spm",
    "unigram",
    "unigram5000_units.txt",
)

class ModalityDropout(torch.nn.Module):
    def __init__(self, p):  # probability of dropping audio
        super().__init__()
        self.p = p

    def forward(self, x,type):
        if torch.rand(1).item() < self.p and type!="RealVideo-FakeAudio":
            return torch.zeros_like(x),torch.tensor(1, dtype=torch.long, requires_grad=False) #returns new tensor that matches x's properties
        return x,torch.tensor(0, dtype=torch.long, requires_grad=False)


class FunctionalModule(torch.nn.Module):
    def __init__(self, functional):
        super().__init__()
        self.functional = functional

    def forward(self, input):
        return self.functional(input)


class AdaptiveTimeMask(torch.nn.Module):
    """
    window-> maximum lenght of mask , choosen randomly bw 0 and window 
    stride-> interval between two masks , i.e -> AdaptiveTimeMask(6400, 16000) means , max of 6400 samples will be masked every 16000 samples
    """
    def __init__(self, window, stride):
        super().__init__()
        self.window = window
        self.stride = stride

    def forward(self, x):
        # x: [T, ...]
        cloned = x.clone()
        length = cloned.size(0)
        n_mask = int((length + self.stride - 0.1) // self.stride)
        ts = torch.randint(0, self.window, size=(n_mask, 2))
        for t, t_end in ts:
            if length - t <= 0:
                continue
            t_start = random.randrange(0, length - t)
            if t_start == t_start + t:
                continue
            t_end += t_start
            cloned[t_start:t_end] = 0
        return cloned



class AddNoise(torch.nn.Module):
    def __init__(
        self,
        noise_filename=NOISE_FILENAME,
        snr_target=None,
    ):
        super().__init__()
        self.snr_levels = [snr_target] if snr_target else [-5, 0, 5, 10, 15, 20, 999999]
        self.noise, sample_rate = torchaudio.load(noise_filename)
        assert sample_rate == 16000

    def forward(self, speech):
        # speech: T x 1
        # return: T x 1
        speech = speech.t()
        start_idx = random.randint(0, self.noise.shape[1] - speech.shape[1])
        noise_segment = self.noise[:, start_idx : start_idx + speech.shape[1]]
        snr_level = torch.tensor([random.choice(self.snr_levels)])
        noisy_speech = torchaudio.functional.add_noise(speech, noise_segment, snr_level)
        return noisy_speech.t()


class VideoTransform:
    def __init__(self, subset):
        if subset == "train":
            self.video_pipeline = torch.nn.Sequential(
                FunctionalModule(lambda x: x / 255.0),
                torchvision.transforms.RandomCrop(88),
                torchvision.transforms.Grayscale(),
                torchvision.transforms.RandomHorizontalFlip(p=0.5),
                AdaptiveTimeMask(5, 25),
                torchvision.transforms.Normalize(0.421, 0.165),
            )
        elif subset == "val" or subset == "test":
            self.video_pipeline = torch.nn.Sequential(
                FunctionalModule(lambda x: x / 255.0),
                torchvision.transforms.CenterCrop(88),
                torchvision.transforms.Grayscale(),
                torchvision.transforms.Normalize(0.421, 0.165),
            )

    def __call__(self, sample):
        # sample: T x C x H x W
        # rtype: T x 1 x H x W
        return self.video_pipeline(sample)

class AudioTransform:
    """1
    It returns a flag too , keep in mind while using it
    flag->1 means audio is dropped -> return audio tensor with all zeroes
    flag->0 means audio is not dropped
    """
    def __init__(self, subset, snr_target=None,modality_drop_rate=None):
        if subset == "train":
            self.time_mask = AdaptiveTimeMask(3200, 16000)
            self.dropout = ModalityDropout(p=modality_drop_rate)   # will output (x, flag)
            self.noise = AddNoise()
            self.norm = FunctionalModule(
                lambda x: torch.nn.functional.layer_norm(x, x.shape, eps=1e-8)
            )
        elif subset == "val":
            self.noise = AddNoise(snr_target=snr_target) if snr_target is not None else None
            # self.dropout = ModalityDropout(p=modality_drop_rate)
            self.norm = FunctionalModule(
                lambda x: torch.nn.functional.layer_norm(x, x.shape, eps=1e-8)
            )
            self.dropout = None
            self.time_mask = None
        elif subset =='test':
            self.noise = AddNoise(snr_target=snr_target) if snr_target is not None else None
            self.dropout = ModalityDropout(p=modality_drop_rate)
            self.norm = FunctionalModule(
                lambda x: torch.nn.functional.layer_norm(x, x.shape, eps=1e-8)
            )
            self.time_mask = None

    def __call__(self, sample,type):
        # sample: T x 1
        flag=torch.tensor(0, dtype=torch.long, requires_grad=False)
        if self.time_mask:
            sample = self.time_mask(sample)

        if self.dropout:
            sample, flag = self.dropout(sample,type)
        # if self.noise:
        #     sample = self.noise(sample)
        sample = self.norm(sample)
        return sample, flag
    

class TextTransform:
    """Mapping Dictionary Class for SentencePiece tokenization."""

    def __init__(
        self,
        sp_model_path=SP_MODEL_PATH,
        dict_path=DICT_PATH,
    ):

        # Load SentencePiece model
        self.spm = sentencepiece.SentencePieceProcessor(model_file=sp_model_path)

        # Load units and create dictionary
        units = open(dict_path, encoding='utf8').read().splitlines()
        self.hashmap = {unit.split()[0]: unit.split()[-1] for unit in units}
        # 0 will be used for "blank" in CTC
        self.token_list = ["<blank>"] + list(self.hashmap.keys()) + ["<eos>"]
        self.ignore_id = -1

    def tokenize(self, text):
        tokens = self.spm.EncodeAsPieces(text)
        token_ids = [self.hashmap.get(token, self.hashmap["<unk>"]) for token in tokens]
        return torch.tensor(list(map(int, token_ids)))

    def post_process(self, token_ids):
        token_ids = token_ids[token_ids != -1]
        text = self._ids_to_str(token_ids, self.token_list)
        text = text.replace("\u2581", " ").strip()
        return text

    def _ids_to_str(self, token_ids, char_list):
        token_as_list = [char_list[idx] for idx in token_ids]
        return "".join(token_as_list).replace("<space>", " ")
    


if __name__ == "__main__":
    audio = torch.randn((69632,1))
    audio_transform = AudioTransform("train")

    out = audio_transform(audio)
    print(out[0].shape,out[1])
    print(out[1].item())  