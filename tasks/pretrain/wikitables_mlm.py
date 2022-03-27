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
"""The WikiTable pretrain corpus for MLM pretrain task."""
import ast
import csv
import os
import json
import pickle
from typing import Dict, List, Union, Any

import datasets

# from utils import WikiTableSample, Table, Question, Cell, Date

# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """\

"""

_DESCRIPTION = """\
The WikiTable pretrain corpus for MLM pretrain task.
"""

_HOMEPAGE = ""

_LICENSE = "CC-BY-SA-4.0 License"

_URL = ""

_LOCAL_PATH = "/mnt/zhoujun/tapas-master/tapas/datasets/pretrain_corpus/gcp_output/"


# WikiTable Dataset
class WikiTable(datasets.GeneratorBasedBuilder):
    """The WikiTable pretrain corpus."""

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(  # temporarily follow wtq pattern
                {
                    "id": datasets.Value("string"),
                    "question": {"title": datasets.Value("string"),
                                 "description": datasets.Value("string")},
                    "table": {"header": datasets.features.Sequence(datasets.Value("string")),
                              "rows": datasets.features.Sequence(datasets.features.Sequence(datasets.Value("string")))},
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # data_dir = os.path.join(dl_manager.download_and_extract(_URL))
        data_dir = _LOCAL_PATH

        return [
            # TODO: set the path from outside
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": os.path.join(data_dir, "wikitables_dump_1000000.pt"), "data_dir": data_dir},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": os.path.join(data_dir, "wikitables_dump_dev.pt"), "data_dir": data_dir},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": os.path.join(data_dir, "wikitables_dump_test.pt"), "data_dir": data_dir}

            )
        ]

    def _generate_examples(self, filepath, data_dir):
        """Yields examples."""
        idx = 0
        with open(filepath, 'rb') as f:
            while True:
                try:
                    raw_samples = pickle.load(f)
                    for raw_sample in raw_samples:
                        table, questions = raw_sample.table, raw_sample.questions
                        title = questions.get('TITLE', '')
                        description = questions.get('DESCRIPTION', '')
                        yield idx, {
                            "id": raw_sample.id,
                            "question": {
                                "title": title.original_text if title else '',
                                "description": description.original_text if description else ''
                            },
                            "table": {
                                "header": [cell.text for cell in table.columns],
                                "rows": [[cell.text for cell in row] for row in table.rows]
                            }
                        }
                        idx += 1
                except EOFError:
                    break
