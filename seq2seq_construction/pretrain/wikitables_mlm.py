import copy
import os
from copy import deepcopy

import torch
from datasets import DatasetDict
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from transformers import AutoTokenizer, AutoConfig
from tqdm import tqdm
from utils.processor import get_default_processor, get_pretrain_wikitable_processor
from utils import WikiTableSample


class Constructor(object):
    def __init__(self, args):
        self.args = args

    def to_seq2seq(self, raw_datasets: DatasetDict, cache_root: str):
        if not len(raw_datasets) == 3:
            raise AssertionError("Train, Dev, Test sections of dataset expected.")
        train_dataset = TrainDataset(self.args, raw_datasets['train'], cache_root)
        dev_dataset = DevDataset(self.args, raw_datasets['validation'], cache_root)
        test_dataset = TestDataset(self.args, raw_datasets['test'], cache_root)

        return train_dataset, dev_dataset, test_dataset


"""
    Raw data are formatted as:
    {
        "id": datasets.Value("string"),
        "question": {"title": datasets.Value("string"),
                     "description": datasets.Value("string")},
        "table_id": datasets.Value("string"),
        "table": {"header": datasets.features.Sequence(datasets.Value("string")),
                  "rows": datasets.features.Sequence(datasets.features.Sequence(datasets.Value("string")))},
    }
    """


class TrainDataset(Dataset):

    def __init__(self, args, raw_datasets, cache_root):
        # This tab processor is for table truncation and linearize.
        self.tokenizer = AutoTokenizer.from_pretrained(args.bert.location, use_fast=False)
        if args.special_tokens:
            self.tokenizer.add_tokens([v for k, v in args.special_tokens])
        self.raw_datasets = raw_datasets

        cache_path = os.path.join(cache_root, args.dataset.train_cache_file)
        if os.path.exists(cache_path) and args.dataset.use_cache:
            print(f"Loading train from cache {cache_path} .")
            self.extended_data = torch.load(cache_path)
        else:
            self.tab_processor = get_pretrain_wikitable_processor(
                args=args,
                tokenizer=self.tokenizer,
                model_config=AutoConfig.from_pretrained(args.bert.location)
            )

            self.extended_data = []
            expansion = args.seq2seq.expansion if args.seq2seq.expansion else 1
            # cnt = 0
            for expand_id in range(expansion):
                for raw_data in tqdm(self.raw_datasets):
                    extend_data = deepcopy(raw_data)
                    question = extend_data["question"]
                    question = f"title : {question['title']} ; description : {question['description']} "
                    table = extend_data['table']

                    table_content = copy.deepcopy(table)
                    try:
                        item_tensor_dict = self.tab_processor.process(
                            table_content=table_content,
                            question=question
                        )
                        # item_tensor_dict['seq_out'] \
                        #     = self.tokenizer.convert_tokens_to_string(item_tensor_dict['output_tokens'])
                        # item_tensor_dict['seq_in'] \
                        #     = self.tokenizer.convert_tokens_to_string(item_tensor_dict['input_tokens'])
                        self.extended_data.append(item_tensor_dict)
                    except Exception as e:
                        print(f"Skip the example because {e}")

                    # # pretrain to tensor (debug)    # TODO: tmp for debug
                    # ff = open('./debug.txt', 'a')
                    # print("seq_in->", file=ff)
                    # print(item_tensor_dict['seq_in'], file=ff)
                    # print("seq_out->", file=ff)
                    # print(item_tensor_dict['seq_out'], file=ff)
                    # input_ids, input_tokens = item_tensor_dict['input_ids'], item_tensor_dict['input_tokens']
                    # attention_mask = item_tensor_dict['attention_mask']
                    # labels, output_tokens = item_tensor_dict['labels'], item_tensor_dict['output_tokens']
                    # input_row_ids, input_column_ids = item_tensor_dict['input_row_ids'], item_tensor_dict[
                    #     'input_column_ids']
                    # output_row_ids, output_column_ids = item_tensor_dict['output_row_ids'], item_tensor_dict[
                    #     'output_column_ids']
                    # for idx, input_id in enumerate(input_ids, 0):
                    #     input_token = self.tokenizer.convert_ids_to_tokens([input_id])
                    #     print(input_id, input_token, input_row_ids[idx], input_column_ids[idx], "attn_mask:",
                    #           attention_mask[idx], file=ff)
                    # print('-' * 80, file=ff)
                    # for idx, output_id in enumerate(labels, 0):
                    #     if output_id == -100:
                    #         output_token = '<p>'
                    #     else:
                    #         output_token = self.tokenizer.convert_ids_to_tokens([output_id])
                    #     print(output_id, output_token, output_row_ids[idx], output_column_ids[idx], file=ff)
                    #

                    # cnt += 1
                    # if cnt > 256:
                    #     break
            if args.dataset.use_cache:
                torch.save(self.extended_data, cache_path)

    def __getitem__(self, index) -> T_co:
        return self.extended_data[index]

    def __len__(self):
        return len(self.extended_data)


class DevDataset(Dataset):

    def __init__(self, args, raw_datasets, cache_root):
        # This tab processor is for table truncation and linearize.
        self.tokenizer = AutoTokenizer.from_pretrained(args.bert.location, use_fast=False)
        if args.special_tokens:
            self.tokenizer.add_tokens([v for k, v in args.special_tokens])
        self.raw_datasets = raw_datasets

        cache_path = os.path.join(cache_root, args.dataset.dev_cache_file)
        if os.path.exists(cache_path) and args.dataset.use_cache:
            print(f"Loading dev from cache {cache_path} .")
            self.extended_data = torch.load(cache_path)
        else:
            self.tab_processor = get_pretrain_wikitable_processor(
                args=args,
                tokenizer=self.tokenizer,
                model_config=AutoConfig.from_pretrained(args.bert.location)
            )

            self.extended_data = []
            expansion = args.seq2seq.expansion if args.seq2seq.expansion else 1
            for expand_id in range(expansion):
                for raw_data in tqdm(self.raw_datasets):
                    extend_data = deepcopy(raw_data)
                    question = extend_data["question"]
                    question = f"title : {question['title']} ; description : {question['description']} "
                    table = extend_data['table']

                    table_content = copy.deepcopy(table)
                    try:
                        item_tensor_dict = self.tab_processor.process(
                            table_content=table_content,
                            question=question
                        )
                        item_tensor_dict['seq_out'] \
                            = self.tokenizer.convert_tokens_to_string(item_tensor_dict['output_tokens'])
                        item_tensor_dict['seq_in'] \
                            = self.tokenizer.convert_tokens_to_string(item_tensor_dict['input_tokens'])
                        self.extended_data.append(item_tensor_dict)
                    except Exception as e:
                        print(f"Skip the example because {e}")
            if args.dataset.use_cache:
                torch.save(self.extended_data, cache_path)

    def __getitem__(self, index) -> T_co:
        return self.extended_data[index]

    def __len__(self):
        return len(self.extended_data)


class TestDataset(Dataset):

    def __init__(self, args, raw_datasets, cache_root):
        # This tab processor is for table truncation and linearize.
        self.tokenizer = AutoTokenizer.from_pretrained(args.bert.location, use_fast=False)
        if args.special_tokens:
            self.tokenizer.add_tokens([v for k, v in args.special_tokens])
        self.raw_datasets = raw_datasets

        cache_path = os.path.join(cache_root, args.dataset.test_cache_file)
        if os.path.exists(cache_path) and args.dataset.use_cache:
            print(f"Loading test from cache {cache_path} .")
            self.extended_data = torch.load(cache_path)
        else:
            self.tab_processor = get_pretrain_wikitable_processor(
                args=args,
                tokenizer=self.tokenizer,
                model_config=AutoConfig.from_pretrained(args.bert.location)
            )

            self.extended_data = []
            expansion = args.seq2seq.expansion if args.seq2seq.expansion else 1
            for expand_id in range(expansion):
                for raw_data in tqdm(self.raw_datasets):
                    extend_data = deepcopy(raw_data)
                    question = extend_data["question"]
                    question = f"title : {question['title']} ; description : {question['description']} "
                    table = extend_data['table']

                    table_content = copy.deepcopy(table)
                    try:
                        item_tensor_dict = self.tab_processor.process(
                            table_content=table_content,
                            question=question
                        )
                        item_tensor_dict['seq_out'] \
                            = self.tokenizer.convert_tokens_to_string(item_tensor_dict['output_tokens'])
                        item_tensor_dict['seq_in'] \
                            = self.tokenizer.convert_tokens_to_string(item_tensor_dict['input_tokens'])
                        self.extended_data.append(item_tensor_dict)
                    except Exception as e:
                        print(f"Skip the example because {e}")
            if args.dataset.use_cache:
                torch.save(self.extended_data, cache_path)

    def __getitem__(self, index) -> T_co:
        return self.extended_data[index]

    def __len__(self):
        return len(self.extended_data)
