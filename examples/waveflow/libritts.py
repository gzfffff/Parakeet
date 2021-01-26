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
import paddle
import random
from paddle.io import Dataset, DataLoader
from parakeet.data.batch import batch_spec, batch_wav


# from parakeet.data.batch import batch_spec, batch_wav
# from parakeet.data import dataset
# from parakeet.audio import AudioProcessor

class LibriTTS(Dataset):
    def __init__(self, root):
        self.root = Path(root).expanduser()
        meta_data = pandas.read_csv(
            # TODO
            str(self.root / "mixture_100_360.csv"),
            sep="\t",
            header=None,
            names=['fname', 'speaker_id', 'chapter_id', 'text', 'frames', 'samples'])
        speaker_ids = list({item.speaker_id for item in meta_data.itertuples()})

        speaker_utterances = {}
        for row in meta_data.itertuples():
            # print(row)
            # 长度小于等于65帧则直接忽略
            if row.frames <= 65:
                continue
            speaker_id = row.speaker_id
            if speaker_id not in speaker_utterances:
                # speaker_utterances.update({speaker_id: []})
                speaker_utterances[speaker_id] = []
            mel_path = str(self.root / "mel" / (row.fname + ".npy"))
            wav_path = str(self.root / "wav" / (row.fname + ".npy"))
            speaker_utterances[speaker_id].append([mel_path, wav_path])
        self.speaker_ids = speaker_ids
        self.speaker_utterances = speaker_utterances
        # print("finish loading metadata.csv!")

    def __getitem__(self, utterence):
        mel_path, wav_path = utterence
        mel = np.load(mel_path)
        wav = np.load(wav_path)
        # print("mel_path: {}, wav_path: {}".format(mel_path, wav_path))
        # print('mel: {}, wav: {}'.format(mel.shape, wav.shape))
        return mel, wav
        # return mel_path, wav_path
        # return np.random.rand(80, 130), np.random.rand(33280)

    def __len__(self):
        return sum([len(utterances) for utterances in self.speaker_utterances.values()])


class LibriTTSVal(Dataset):
    """A simple dataset adaptor for the processed ljspeech dataset."""

    def __init__(self, root):
        self.root = Path(root).expanduser()
        meta_data = pandas.read_csv(
            str(self.root / "validation.csv"),
            sep="\t",
            header=None,
            names=['speaker', 'fname'])

        records = []
        for row in meta_data.itertuples():
            mel_path = str(self.root / "mel" / (row.fname + ".npy"))
            wav_path = str(self.root / "wav" / (row.fname + ".npy"))
            records.append((mel_path, wav_path))
        self.records = records

    def __getitem__(self, i):
        mel_name, wav_name = self.records[i]
        mel = np.load(mel_name)
        wav = np.load(wav_name)
        return mel, wav

    def __len__(self):
        return len(self.records)



class SpeakerSampler(paddle.io.BatchSampler):
    def __init__(self, dataset: LibriTTS, batch_size):
        self._speakers = dataset.speaker_ids
        self._speaker_utterances = dataset.speaker_utterances

        self.batch_size = batch_size

    def __iter__(self):
        speaker_generator = iter(random_cycle(self._speakers))
        speaker_utterances_generator = {s: iter(random_cycle(us)) for s, us in self._speaker_utterances.items()}

        # utterances = []
        while True:
            speaker = next(speaker_generator)

            utterances = []
            for _ in range(self.batch_size):
                utterances.append(next(speaker_utterances_generator[speaker]))
            # print(utterances)
            yield utterances


def random_cycle(iterable):
    # cycle('ABCD') --> A B C D B C D A A D B C ...
    saved = []
    for element in iterable:
        yield element
        saved.append(element)
    random.shuffle(saved)
    while saved:
        for element in saved:
              yield element
        random.shuffle(saved)


class TestSampler(paddle.io.BatchSampler):
    def __init__(self, dataset: LibriTTS, batch_size):
        self.batch_size = batch_size

    def __iter__(self):
        while True:
            batch = [[i, i + 1] for i in range(self.batch_size)]
            print(batch)
            yield batch


class LibriTTSClipCollector(object):
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
        try:
            mel, wav = example
            frames = mel.shape[-1]
            start = np.random.randint(0, frames - self.clip_frames)
            mel_clip = mel[:, start:start + self.clip_frames]
            wav_clip = wav[start * self.hop_length:(start + self.clip_frames) * self.hop_length]
        except Exception as e:
            print('-----'*10 + 'LibriTTSClipCollector raises a exception' + '-----'*10)
            print(repr(e))
        return mel_clip, wav_clip


class LibriTTSCollector(object):
    """A simple callable to batch LJSpeech examples."""

    def __init__(self, padding_value=0.):
        self.padding_value = padding_value

    def __call__(self, examples):
        mels = [example[0] for example in examples]
        wavs = [example[1] for example in examples]
        mels = batch_spec(mels, pad_value=self.padding_value)
        wavs = batch_wav(wavs, pad_value=self.padding_value)
        return mels, wavs



if __name__ == '__main__':
    from paddle.io import DataLoader
    # from ljspeech import LJSpeech, LJSpeechClipCollector, LJSpeechCollector

    # batch_fn = LJSpeechClipCollector(65, 256)
    root = "/extra_mnt/audio_datasets/libritts/libritts/LibriTTS/libritts_waveflow/train-clean-100/"
    val_root = "/extra_mnt/audio_datasets/libritts/libritts/LibriTTS/libritts_waveflow/train-clean-100/"
    batch_size = 6
    libritts_dataset = LibriTTS(root)
    libritts_sampler = SpeakerSampler(libritts_dataset, batch_size)
    test_sampler = TestSampler(libritts_dataset, batch_size)
    collate_fn =  LibriTTSClipCollector()

    print(libritts_dataset.speaker_utterances.keys())

    # iterator = iter(libritts_sampler)
    # for i, item in enumerate(iterator):
    #     if i > 1000:
    #         break
    #     if i % 100 == 0:
    #         print(item)


    print('-----'*10)
    '''
    print(len(libritts_dataset))
    dataloader = DataLoader(libritts_dataset, batch_sampler=libritts_sampler, num_workers=0, collate_fn=collate_fn)
    for i, batch in enumerate(dataloader):
        if i % 1000 == 0:
            print("batch: {}, {}".format(batch[0].shape, batch[1].shape))
        # print("batch: {}, {}".format(batch[0].shape, batch[1].shape))
        # i = 0
    '''


    # validation dataset test
    val_libritts_dataset = LibriTTSVal(val_root)
    val_collate_fn = LibriTTSCollector()
    val_dataloader = DataLoader(val_libritts_dataset, 
                                batch_size=1,
                                shuffle=True,
                                drop_last=True,
                                collate_fn=val_collate_fn)
    for i, batch in enumerate(val_dataloader):
        if i % 20 == 0:
            print("batch: {}, {}".format(batch[0].shape, batch[1].shape))





