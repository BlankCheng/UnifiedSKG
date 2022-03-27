# coding=utf-8
# Copyright 2021 The HuggingFace Datasets Authors, The Google AI Language Team Authors and the current dataset script contributor.
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
"""The HiTab dataset for table qa over hierarchical tables."""
import ast
import csv
import os
import json

import datasets

# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """\
@article{cheng2021hitab,
  title={HiTab: A Hierarchical Table Dataset for Question Answering and Natural Language Generation},
  author={Cheng, Zhoujun and Dong, Haoyu and Wang, Zhiruo and Jia, Ran and Guo, Jiaqi and Gao, Yan and Han, Shi and Lou, Jian-Guang and Zhang, Dongmei},
  journal={arXiv preprint arXiv:2108.06712},
  year={2021}
}
"""

_DESCRIPTION = """\
HiTab is a hierarchical table dataset for table question answering and data2text.
"""

_HOMEPAGE = "https://github.com/microsoft/HiTab"

_LICENSE = "CC-BY-SA-4.0 License"

_URL = "https://drive.google.com/u/0/uc?id=15825NnuKWGY3D4NHX6BLRrBLc-2GKpKN&export=download"


class HiTab(datasets.GeneratorBasedBuilder):
    """The HiTab dataset"""

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(  # temporarily follow wtq pattern
                {
                    "id": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "table_id": datasets.Value("string"),
                    "table": {"header": datasets.features.Sequence(datasets.Value("string")),
                              "rows": datasets.features.Sequence(datasets.features.Sequence(datasets.Value("string")))},
                    "answer": datasets.features.Sequence(datasets.Value("string")),
                }
            ),
            # features=datasets.Features(
            #     {
            #         "id": datasets.Value("string"),
            #         "question": datasets.Value("string"),
            #         "table_id": datasets.Value("string"),
            #         "table": {
            #             "cells": datasets.features.Sequence(datasets.features.Sequence(datasets.Value("string"))),
            #             "top_header_rows_num": datasets.Value("int32"),
            #             "left_header_columns_num": datasets.Value("int32")
            #         },
            #         "answer": datasets.features.Sequence(datasets.Value("string")),
            #     }
            # ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        data_dir = os.path.join(dl_manager.download_and_extract(_URL))

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": os.path.join(data_dir, "data/train_samples.jsonl"), "data_dir": data_dir},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": os.path.join(data_dir, "data/dev_samples.jsonl"), "data_dir": data_dir},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": os.path.join(data_dir, "data/test_samples.jsonl"),
                            "data_dir": data_dir},
            ),

        ]

    def _generate_examples(self, filepath, data_dir):
        """Yields examples."""
        with open(filepath, encoding='utf-8') as f:
            for idx, line in enumerate(f):
                raw_sample = json.loads(line)
                with open(os.path.join(data_dir, 'data/tables/raw/', f"{raw_sample['table_id']}.json")) as tf:
                    table = json.load(tf)
                yield idx, {
                    "id": raw_sample['id'],
                    "question": raw_sample['question'],
                    "table": {  # temporarily follow wtq pattern
                        "header": table['texts'][0],
                        "rows": table['texts'][1:]
                    },
                    # "table": {
                    #     "cells": table['texts'],
                    #     "top_header_rows_num": table["top_header_rows_num"],
                    #     "left_header_columns_num": table['left_header_columns_num']
                    # },
                    "table_id": raw_sample['table_id'],
                    "answer": [str(a) for a in raw_sample['answer']]
                }
