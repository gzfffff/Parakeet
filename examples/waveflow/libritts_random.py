# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import os
from pathlib import Path
import pickle
import numpy as np
import pandas
import random
from paddle.io import Dataset, DataLoader

from parakeet.data.batch import batch_spec, batch_wav
from parakeet.data import dataset
from parakeet.audio import AudioProcessor


class LibriTTS(Dataset):
    """A simple dataset adaptor for the processed ljspeech dataset."""

    def __init__(self, root):
        self.root = Path(root).expanduser()
        meta_data = pandas.read_csv(
            str(self.root / "metadata.csv"),
            sep="\t",
            header=None,
            names=['fname', 'speaker_id', 'chapter_id', 'text', 'frames', 'samples'])

        records = []
        for row in meta_data.itertuples():
            if row.frames <= 65:
                continue
            mel_path = str(self.root / "mel" / (row.fname + ".npy"))
            wav_path = str(self.root / "wav" / (row.fname + ".npy"))
            records.append((mel_path, wav_path))
        self.records = records
        random.shuffle(self.records)

    def __getitem__(self, i):
        mel_name, wav_name = self.records[i]
        mel = np.load(mel_name)
        wav = np.load(wav_name)
        return mel, wav

    def __len__(self):
        return len(self.records)


class LibriTTSCollector(object):
    """A simple callable to batch LJSpeech examples."""

    def __init__(self, padding_value=0.):
        self.padding_value = padding_value

    def __call__(self, examples):
        mels = [example[0] for example in examples]
        wavs = [example[1] for example in examples]
        # 对一个batch进行pad
        # mels: [batch, 80, n_frames]
        # wavs: [batch, time_len]
        mels = batch_spec(mels, pad_value=self.padding_value)
        wavs = batch_wav(wavs, pad_value=self.padding_value)
        return mels, wavs


class LibriTTSClipCollector(object):
    '''
        对序列进行裁剪，使其长度为65帧，65 * 256 = 16640
    '''
    def __init__(self, clip_frames=65, hop_length=256):
        self.clip_frames = clip_frames
        self.hop_length = hop_length

    def __call__(self, examples):
        mels = []
        wavs = []
        for example in examples:
            mel_clip, wav_clip = self.clip(example)
            mels.append(mel_clip)
            wavs.append(wav_clip)
        mels = np.stack(mels)
        wavs = np.stack(wavs)
        return mels, wavs

    def clip(self, example):
        # 随机选择长度为65帧的mel和16640长度的wav
        mel, wav = example
        frames = mel.shape[-1]
        start = np.random.randint(0, frames - self.clip_frames)
        mel_clip = mel[:, start:start + self.clip_frames]
        wav_clip = wav[start * self.hop_length:(start + self.clip_frames) *
                       self.hop_length]
        return mel_clip, wav_clip
