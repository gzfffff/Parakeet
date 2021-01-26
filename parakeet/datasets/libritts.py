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

from paddle.io import Dataset
from pathlib import Path
import os

__all__ = ["LibriTTSMetaData"]


class LibriTTSMetaData(Dataset):
    def __init__(self, root):
        """
        :param root: libritts数据集的路径
        """
        self.root = os.path.abspath(root)
        records = []
        index = 1
        self.meta_info = ["index", "file_path", "speaker_id", "chapter_id", "text"]
        for root, dirs, files in os.walk(self.root):
            # root: 当前目录路径 dirs: 当前路径下所有子目录 files: 当前路径下所有非目录子文件
            if len(files) == 0:
                continue
            for filename in files:
                if filename.endswith(".wav"):
                    file_path = os.path.join(root, filename)
                    # info: [speaker_id, chapter_id, index, "000000"]
                    filename_without_shuffix = os.path.splitext(filename)[0]
                    info = filename_without_shuffix.split('_')
                    speaker_id = info[0]
                    chapter_id = info[1]
                    text_file_path = os.path.join(root, filename_without_shuffix + ".normalized.txt")
                    if not os.path.isfile(text_file_path):
                        continue
                    with open(text_file_path, "rt", encoding="utf-8") as f:
                        normalized_text = f.readlines()[0].strip()
                    records.append([index, file_path, speaker_id, chapter_id, normalized_text])
                    index += 1
        self.records = records
        '''
        self.root = Path(root).expanduser()
        wav_dir = self.root / "wavs"
        csv_path = self.root / "metadata.csv"
        records = []
        speaker_name = "ljspeech"
        with open(str(csv_path), 'rt', encoding='utf-8') as f:
            for line in f:
                filename, _, normalized_text = line.strip().split("|")
                filename = str(wav_dir / (filename + ".wav"))
                records.append([filename, normalized_text, speaker_name])
        self.records = records
        '''

    def __getitem__(self, i):
        return self.records[i]

    def __len__(self):
        return len(self.records)

    def get_meta_info(self):
        return self.meta_info


