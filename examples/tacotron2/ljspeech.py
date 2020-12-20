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
from paddle.io import Dataset, DataLoader

from parakeet.data.batch import batch_spec, batch_text_id
from parakeet.data import dataset


class LJSpeech(Dataset):
    """A simple dataset adaptor for the processed ljspeech dataset."""

    def __init__(self, root):
        self.root = Path(root).expanduser()
        records = []
        with open(self.root / "metadata.pkl", 'rb') as f:
            metadata = pickle.load(f)
        for mel_name, text, ids in metadata:
            mel_name = self.root / "mel" / (mel_name + ".npy")
            records.append((mel_name, text, ids))
        self.records = records

    def __getitem__(self, i):
        mel_name, _, ids = self.records[i]
        mel = np.load(mel_name)
        return ids, mel

    def __len__(self):
        return len(self.records)


class LJSpeechCollector(object):
    """A simple callable to batch LJSpeech examples."""

    def __init__(self, padding_idx=0, padding_value=0.,
                 padding_stop_token=1.0):
        self.padding_idx = padding_idx
        self.padding_value = padding_value
        self.padding_stop_token = padding_stop_token

    def __call__(self, examples):
        texts = []
        mels = []
        text_lens = []
        mel_lens = []
        stop_tokens = []
        for data in examples:
            text, mel = data
            text = np.array(text, dtype=np.int64)
            text_lens.append(len(text))
            mels.append(mel)
            texts.append(text)
            mel_lens.append(mel.shape[1])
            stop_token = np.zeros([mel.shape[1] - 1], dtype=np.float32)
            stop_tokens.append(np.append(stop_token, 1.0))

        # Sort by text_len in descending order
        texts = [
            i
            for i, _ in sorted(
                zip(texts, text_lens), key=lambda x: x[1], reverse=True)
        ]
        mels = [
            i
            for i, _ in sorted(
                zip(mels, text_lens), key=lambda x: x[1], reverse=True)
        ]

        mel_lens = [
            i
            for i, _ in sorted(
                zip(mel_lens, text_lens), key=lambda x: x[1], reverse=True)
        ]

        stop_tokens = [
            i
            for i, _ in sorted(
                zip(stop_tokens, text_lens), key=lambda x: x[1], reverse=True)
        ]

        text_lens = sorted(text_lens, reverse=True)

        # Pad sequence with largest len of the batch
        texts = batch_text_id(texts, pad_id=self.padding_idx)
        mels = np.transpose(
            batch_spec(
                mels, pad_value=self.padding_value), axes=(0, 2, 1))
        stop_tokens = batch_text_id(
            stop_tokens, pad_id=self.padding_stop_token, dtype=mels[0].dtype)

        return (texts, mels, text_lens, mel_lens, stop_tokens)