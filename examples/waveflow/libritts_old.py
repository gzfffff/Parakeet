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

# from parakeet.data.batch import batch_spec, batch_wav
# from parakeet.data import dataset
# from parakeet.audio import AudioProcessor

class LibriTTS(Dataset):
    def __init__(self, root):
        self.root = Path(root).expanduser()
        meta_data = pandas.read_csv(
            # TODO
            str(self.root / "metadata.csv"),
            sep="\t",
            header=None,
            names=['fname', 'speaker_id', 'chapter_id', 'text', 'frames', 'samples'])
        speaker_ids = list({item.speaker_id for item in meta_data.itertuples()})

        speaker_utterances = {}
        for row in meta_data.itertuples():
            # print(row)
            speaker_id = row.speaker_id
            if speaker_id not in speaker_utterances:
                # speaker_utterances.update({speaker_id: []})
                speaker_utterances[speaker_id] = []
            mel_path = str(self.root / "mel" / (row.fname + ".npy"))
            wav_path = str(self.root / "wav" / (row.fname + ".npy"))
            speaker_utterances[speaker_id].append([mel_path, wav_path])
        self.speaker_ids = speaker_ids
        self.speaker_utterances = speaker_utterances

    def __getitem__(self, utterence):
        mel_path, wav_path = utterence
        # mel = np.load(mel_path)
        # wav = np.load(wav_path)
        # print("mel_path: {}, wav_path: {}".format(mel_path, wav_path))
        # return mel, wav
        # return mel_path, wav_path
        return np.random.rand(80, 130), np.random.rand(33280)

    def __len__(self):
        return sum([len(utterances) for utterances in self.speaker_utterances.values()])



class SpeakerSampler(paddle.io.BatchSampler):
    def __init__(self, dataset: LibriTTS, batch_size):
        self._speakers = dataset.speaker_ids
        self._speaker_utterances = dataset.speaker_utterances

        self.batch_size = batch_size

    def __iter__(self):
        # utterances = [value for values in self._speaker_utterances.values() for value in values]
        utterances_generator = iter(random_cycle(self._speakers, self._speaker_utterances))
        # utterances = []
        while True:
            utterances = []
            for _ in range(self.batch_size):
                utterances.append(next(utterances_generator))
            # print(utterances)
            yield utterances

class TestSampler(paddle.io.BatchSampler):
    def __init__(self, dataset: LibriTTS, batch_size):
        self.batch_size = batch_size

    def __iter__(self):
        while True:
            batch = [[i, i + 1] for i in range(self.batch_size)]
            print(batch)
            yield batch


def random_cycle(speakers, speaker_utterances):
    # cycle('ABCD') --> A B C D B C D A A D B C ...

    utterances = [value for speaker in speakers for value in speaker_utterances[speaker]]

    """
    print(speakers)
    random.shuffle(speakers)
    print("aaaaaaaaaa")
    for speaker in speaker_utterances:
        random.shuffle(speaker_utterances[speaker])
    print("bbbbbbbbbb")
    utterances = [value for speaker in speakers for value in speaker_utterances[speaker]]
    """

    while True:
        print(len(utterances))
        for utterance in utterances:
              yield utterance
        random.shuffle(speakers)
        for speaker in speaker_utterances:
            random.shuffle(speaker_utterances[speaker])
        utterances = [value for speaker in speakers for value in speaker_utterances[speaker]]


if __name__ == '__main__':
    from paddle.io import DataLoader
    # from ljspeech import LJSpeech, LJSpeechClipCollector, LJSpeechCollector

    # batch_fn = LJSpeechClipCollector(65, 256)
    root = "./"
    batch_size = 6
    libritts_dataset = LibriTTS(root)
    libritts_sampler = SpeakerSampler(libritts_dataset, batch_size)
    test_sampler = TestSampler(libritts_dataset, batch_size)

    print(libritts_dataset.speaker_utterances.keys())

    # iterator = iter(libritts_sampler)
    # for i, item in enumerate(iterator):
    #     if i > 1000:
    #         break
    #     if i % 100 == 0:
    #         print(item)


    print('-----'*10)
    print(len(libritts_dataset))
    dataloader = DataLoader(libritts_dataset, batch_sampler=libritts_sampler, num_workers=0)
    for i, batch in enumerate(dataloader):
        if i % 100 == 0:
            print("batch: {}, {}".format(batch[0].shape, batch[1].shape))
        # i = 0










