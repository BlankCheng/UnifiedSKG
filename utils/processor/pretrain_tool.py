"""
Utils for pretrain processing.
"""
import abc
from typing import Dict, List
import random
import numpy as np

import torch
from transformers import (
    AutoTokenizer,
    BasicTokenizer,
    AutoConfig,
    PretrainedConfig,
    BartTokenizer,
    BartTokenizerFast,
    T5Tokenizer,
    T5TokenizerFast,
)

MASK_TOKEN = '<mask>'


class PretrainTool(object):
    def __init__(self):
        pass


class PretrainMLMToolDeprecated(PretrainTool):
    """
    Masked Language Modeling (MLM) pretrain preprocessing tool.
    """

    def __init__(self,
                 tokenizer: BasicTokenizer = None,
                 mlm_rate: float = 0.15,
                 table_mask_type: str = 'cell',
                 question_mask_type: str = 'token',
                 **kwargs
                 ):
        super(PretrainMLMToolDeprecated, self).__init__()
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="facebook/bart-large")
        else:
            self.tokenizer = tokenizer
        self.mlm_rate = mlm_rate
        self.table_mask_type = table_mask_type
        self.question_mask_type = question_mask_type

        assert self.table_mask_type in ['none', 'token', 'cell', 'column'], \
            f"`{self.table_mask_type}` is not a supported table masking method."
        assert self.question_mask_type in ['none', 'token'], \
            f"`{self.question_mask_type}` is not a supported question masking method."
        # random.seed(2)

    def _not_mask(self, token_list, row_ids, column_ids, **kwargs):
        """
        No mask, just for a unified API.
        """
        return token_list, row_ids, column_ids

    def _simple_token_mask(self, token_list, row_ids=None, column_ids=None, **kwargs):
        """
        A simple token masking function.
        """
        result_token_list = []
        result_row_ids, result_column_ids = [], []
        for idx, token in enumerate(token_list):
            if random.random() <= self.mlm_rate and token not in NOT_MASK_TOKENS:
                mask_tokens = self.tokenizer.tokenize(BART_MASK_TOKEN)
                cell_tokens = mask_tokens
                if row_ids and column_ids:
                    cell_row_ids = [row_ids[idx] for _ in range(len(mask_tokens))]
                    cell_column_ids = [column_ids[idx] for _ in range(len(mask_tokens))]
            else:
                cell_tokens = [token]
                if row_ids and column_ids:
                    cell_row_ids = [row_ids[idx]]
                    cell_column_ids = [column_ids[idx]]

            result_token_list.extend(cell_tokens)
            if row_ids and column_ids:
                result_row_ids.extend(cell_row_ids)
                result_column_ids.extend(cell_column_ids)

        return result_token_list, result_row_ids, result_column_ids

    def _table_cell_mask(
            self,
            token_list,
            table_content,
            row_ids,
            column_ids,
            cell2token_map,
            token2cell_map,
            **kwargs
    ):
        """
        Mask whole cells in tables.
        """
        result_token_list = []
        result_row_ids, result_column_ids = [], []
        idx = 0
        while idx < len(token_list):
            token = token_list[idx]
            if random.random() <= self.mlm_rate \
                    and token not in NOT_MASK_TOKENS \
                    and idx in token2cell_map:  # start token of a cell
                mask_tokens = self.tokenizer.tokenize(BART_MASK_TOKEN)
                cell_tokens = mask_tokens
                if row_ids and column_ids:
                    cell_row_ids = [row_ids[idx] for _ in range(len(mask_tokens))]
                    cell_column_ids = [column_ids[idx] for _ in range(len(mask_tokens))]
                idx = token2cell_map[idx]['end_idx'] + 1
            else:
                cell_tokens = [token]
                if row_ids and column_ids:
                    cell_row_ids = [row_ids[idx]]
                    cell_column_ids = [column_ids[idx]]
                idx += 1

            result_token_list.extend(cell_tokens)
            if row_ids and column_ids:
                result_row_ids.extend(cell_row_ids)
                result_column_ids.extend(cell_column_ids)

        return result_token_list, result_row_ids, result_column_ids

    def _table_column_mask(
            self,
            token_list,
            table_content,
            row_ids,
            column_ids,
            cell2token_map,
            token2cell_map,
            **kwargs
    ):
        """
        Mask header cell(s) and retain its data cells in the same column.
        """
        result_token_list = []
        result_row_ids, result_column_ids = [], []
        idx = 0
        while idx < len(token_list):
            token = token_list[idx]
            if random.random() <= self.mlm_rate \
                    and token not in NOT_MASK_TOKENS \
                    and idx in token2cell_map \
                    and token2cell_map[idx]['coord'][0] == 0:  # the first row, i.e., the column header row
                mask_tokens = self.tokenizer.tokenize(MASK_TOKEN)
                cell_tokens = mask_tokens
                if row_ids and column_ids:
                    cell_row_ids = [row_ids[idx] for _ in range(len(mask_tokens))]
                    cell_column_ids = [column_ids[idx] for _ in range(len(mask_tokens))]
                idx = token2cell_map[idx]['end_idx'] + 1
            else:
                cell_tokens = [token]
                if row_ids and column_ids:
                    cell_row_ids = [row_ids[idx]]
                    cell_column_ids = [column_ids[idx]]
                idx += 1

            result_token_list.extend(cell_tokens)
            if row_ids and column_ids:
                result_row_ids.extend(cell_row_ids)
                result_column_ids.extend(cell_column_ids)

        return result_token_list, result_row_ids, result_column_ids

    def pretrain_process(
            self,
            question,
            linear_table,
            table_content,
            row_ids,
            column_ids,
            cell2token_map,
            token2cell_map
    ):
        """
        Mask `mlm_rate` tokens/cells in the input question and table.
        The pretrain task is to recover both the masked tokens and unmasked tokens, as in BART paper.

        Args
            question (:obj:`str`):
                Question string.
            linear_table (:obj:`str`):
                Linearized table string.
            table_content (:obj:`Dict[str, List]`):
                Raw structured table matrix.
            row_ids (:obj:`List[int]`):
                List of table token-wise row ids.
            column_ids (:obj:`List[int]`):
                List of table token-wise column ids.
            cell2token_map (:obj: `Dict[Tuple, List[int]]`):
                Map that record `coord in table matrix`: [`cell start tok idx`, `cell end tok idx`].
            token2cell_map (:obj: `Dict[int, Dict[str, ]]`):
                Map that records `cell start tok idx`: {`end_idx`: `cell end tok idx`,
                                                        `coord`: (`row_idx`, `col_idx`)}.
        """
        _mask_func_map = {
            'none': self._not_mask,
            'token': self._simple_token_mask,
            'cell': self._table_cell_mask,
            'column': self._table_column_mask
        }

        question_token_list = self.tokenizer.tokenize(question)
        table_token_list = self.tokenizer.tokenize(linear_table)

        # mask question tokens
        question_masked_token_list, _, _ = _mask_func_map[self.question_mask_type](
            token_list=question_token_list
        )
        question_masked = self.tokenizer.convert_tokens_to_string(question_masked_token_list)
        # mask table tokens
        table_masked_token_list, row_ids, column_ids = _mask_func_map[self.table_mask_type](
            token_list=table_token_list,
            table_content=table_content,
            row_ids=row_ids,
            column_ids=column_ids,
            cell2token_map=cell2token_map,
            token2cell_map=token2cell_map
        )
        table_masked = self.tokenizer.convert_tokens_to_string(table_masked_token_list)
        output = question_masked + ' ' + table_masked
        return question_masked, table_masked, output, row_ids, column_ids


class PretrainMLMTool(PretrainTool):
    """
    Masked Language Modeling (MLM) pretrain preprocessing tool.
    """

    def __init__(self,
                 tokenizer: BasicTokenizer = None,
                 mlm_rate: float = 0.15,
                 table_mask_type: str = 'cell',
                 row_sep: str = 'row :',
                 column_sep: str = '|',
                 max_row_size: int = 256,
                 max_column_size: int = 256,
                 **kwargs
                 ):
        super(PretrainMLMTool, self).__init__()
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="facebook/bart-large")
        else:
            self.tokenizer = tokenizer
        self.mlm_rate = mlm_rate
        self.table_mask_type = table_mask_type
        self.row_sep = row_sep.strip()
        self.column_sep = column_sep.strip()
        self.max_row_size = max_row_size
        self.max_column_size = max_column_size

    def _is_column_sep_token(self, token):
        return self.column_sep != '' and self.column_sep in token

    def _build_mask_bit_maps(self, tokens, row_ids, column_ids):
        # traverse to mark start of table tokens and table cell first tokens
        start_table_token_idx, cell_first_end_token_map = None, {}
        last_coord = (self.max_row_size, self.max_column_size)
        idx, next_idx = 0, None
        while idx < len(tokens):
            token, row_id, column_id = tokens[idx], row_ids[idx], column_ids[idx]
            coord = (row_id, column_id)
            if start_table_token_idx is None and row_id != self.max_row_size:
                start_table_token_idx = idx
            if coord != last_coord \
                    and column_id != self.max_column_size \
                    and not self._is_column_sep_token(token):  # cell first token
                next_idx = idx + 1
                while next_idx < len(tokens) \
                        and (row_ids[next_idx], column_ids[next_idx]) == coord \
                        and not self._is_column_sep_token(tokens[next_idx]):
                    next_idx += 1
                cell_first_end_token_map[idx] = next_idx - 1

            last_coord = coord
            if next_idx:
                idx = next_idx
                next_idx = None
            else:
                idx = idx + 1

        # build mask bit maps
        mask_bit_maps = np.zeros((len(tokens),))
        if self.table_mask_type == 'token':
            mask_token_indices = random.sample(
                [i for i in range(len(tokens))],
                int(len(tokens) * self.mlm_rate)
            )
            mask_bit_maps[mask_token_indices] = 1
        else:
            if self.table_mask_type == 'cell':
                candi_cell_token_indices = list(cell_first_end_token_map.keys())
            elif self.table_mask_type == 'column':
                candi_cell_token_indices = [[idx for idx in list(cell_first_end_token_map.keys()) if row_ids[idx] == 0]]
            else:
                raise ValueError(f"{self.table_mask_type} table mask type is not supported. "
                                 f"Please choose from ['token', 'cell', 'column']")
            mask_text_token_indices = random.sample(
                [i for i in range(start_table_token_idx)],
                max(1, int(start_table_token_idx * self.mlm_rate))
            )
            mask_table_token_indices = random.sample(
                candi_cell_token_indices,
                max(1, int(len(candi_cell_token_indices) * self.mlm_rate))
            )
            mask_token_indices = mask_text_token_indices + mask_table_token_indices
            ignore_token_indices = []
            for first_idx in mask_table_token_indices:
                ignore_one_cell_indices = [i for i in range(first_idx+1, cell_first_end_token_map[first_idx]+1)]
                ignore_token_indices.extend(ignore_one_cell_indices)
            mask_bit_maps[mask_token_indices] = 1
            mask_bit_maps[ignore_token_indices] = -1
            # print("start_tok:", tokens[start_table_token_idx], start_table_token_idx)
            # print("mask_text_token_idx")
            # print(mask_text_token_indices)
            # print("mask_table_token_idx")
            # print(mask_table_token_indices)

        return mask_bit_maps

    def pretrain_process(
            self,
            tokens,
            row_ids,
            column_ids,
    ):
        """
        Mask `mlm_rate` tokens/cells in the input question and table.
        The pretrain task is to recover both the masked tokens and unmasked tokens, as in BART/T5 paper.

        Args
            tokens (:obj:`List[str]`):
                Question string.
            row_ids (:obj:`List[int]`):
                List of table token-wise row ids.
            column_ids (:obj:`List[int]`):
                List of table token-wise column ids.
        """
        masked_tokens, masked_row_ids, masked_column_ids = [], [], []
        mask_bit_maps = self._build_mask_bit_maps(tokens, row_ids, column_ids)
        for idx in range(len(tokens)):
            token, row_id, column_id = tokens[idx], row_ids[idx], column_ids[idx]
            if mask_bit_maps[idx] == -1:  # ignore, skip the token since the whole cell is masked
                continue
            elif mask_bit_maps[idx] == 0:  # not mask
                cur_tokens, cur_row_ids, cur_column_ids = [token], [row_id], [column_id]
            elif mask_bit_maps[idx] == 1:  # mask token
                cur_tokens = [MASK_TOKEN]
                cur_row_ids, cur_column_ids = [row_id], [column_id]
            else:
                raise ValueError(f"{mask_bit_maps[idx]} is not supported.")
            masked_tokens.extend(cur_tokens)
            masked_row_ids.extend(cur_row_ids)
            masked_column_ids.extend(cur_column_ids)

        return {
            "input_tokens": masked_tokens,
            "input_row_ids": masked_row_ids,
            "input_column_ids": masked_column_ids,
            "output_tokens": tokens,
            "output_row_ids": row_ids,
            "output_column_ids": column_ids
        }


class PretrainToTensorTool(PretrainTool):
    """
    Build model input and output tensors.
    """

    def __init__(
            self,
            tokenizer: BasicTokenizer,
            model_config: PretrainedConfig,
            max_input_length: int,
            max_generation_length: int,
            max_row_size: int = 256,
            max_column_size: int = 256,
            attention_mask_type: str = "sequence",
    ):
        super(PretrainToTensorTool, self).__init__()
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="facebook/bart-large")
        else:
            self.tokenizer = tokenizer
        if model_config is None:
            self.model_config = AutoConfig.from_pretrained(pretrained_model_name_or_path="facebook/bart-large")
        else:
            self.model_config = model_config
        self.max_input_length = max_input_length
        self.max_generation_length = max_generation_length
        self.max_row_size = max_row_size
        self.max_column_size = max_column_size
        self.attention_mask_type = attention_mask_type

    def pretrain_to_tensor(self, item_dict: Dict):
        # convert tokens to vocab ids, with `bos` and `eos`
        self._to_sequence_ids(item_dict)
        # fill to max length and convert to tensor
        attention_mask = item_dict['attention_mask'] = [1 for _ in item_dict['input_ids']]
        item_dict['input_ids'] = torch.LongTensor(
            self._pad_and_trunc(item_dict['input_ids'], self.model_config.pad_token_id, self.model_config.eos_token_id,
                                self.max_input_length)
        )
        item_dict['attention_mask'] = torch.LongTensor(
            self._pad_and_trunc(attention_mask, 0, 1, self.max_input_length)
        )
        item_dict['labels'] = torch.LongTensor(
            self._pad_and_trunc(item_dict['labels'], -100, self.model_config.eos_token_id, self.max_generation_length)
        )
        item_dict['input_row_ids'] = torch.LongTensor(
            self._pad_and_trunc(item_dict['input_row_ids'], self.max_row_size, self.max_row_size, self.max_input_length)
        )
        item_dict['input_column_ids'] = torch.LongTensor(
            self._pad_and_trunc(item_dict['input_column_ids'], self.max_column_size, self.max_column_size,
                                self.max_input_length)
        )
        item_dict['output_row_ids'] = torch.LongTensor(
            self._pad_and_trunc(item_dict['output_row_ids'], self.max_row_size, self.max_row_size,
                                self.max_generation_length)
        )
        item_dict['output_column_ids'] = torch.LongTensor(
            self._pad_and_trunc(item_dict['output_column_ids'], self.max_column_size, self.max_column_size,
                                self.max_generation_length)
        )
        # refine special `attention_mask`
        self._build_attention_mask(item_dict)

        return item_dict

    def _to_sequence_ids(self, item_dict: Dict):
        if isinstance(self.tokenizer, BartTokenizer) or isinstance(self.tokenizer, BartTokenizerFast):
            bos_token_ids, eos_token_ids = [self.model_config.bos_token_id], [self.model_config.eos_token_id]
        elif isinstance(self.tokenizer, T5Tokenizer) or isinstance(self.tokenizer, T5TokenizerFast):
            bos_token_ids, eos_token_ids = [], [self.model_config.eos_token_id]
        else:
            raise ValueError(f"{type(self.tokenizer)} is currently not supported.")
        item_dict['input_ids'] = self.tokenizer.convert_tokens_to_ids(item_dict['input_tokens'])
        item_dict['labels'] = self.tokenizer.convert_tokens_to_ids(item_dict['output_tokens'])
        item_dict['input_ids'] = bos_token_ids + item_dict['input_ids'] + eos_token_ids
        item_dict['labels'] = bos_token_ids + item_dict['labels'] + eos_token_ids

        bos_row_ids, eos_row_ids = [self.max_row_size for _ in bos_token_ids], [self.max_row_size for _ in
                                                                                eos_token_ids]
        bos_column_ids, eos_column_ids = [self.max_column_size for _ in bos_token_ids], [self.max_column_size for _ in
                                                                                         eos_token_ids]
        item_dict['input_row_ids'] = bos_row_ids + item_dict['input_row_ids'] + eos_row_ids
        item_dict['input_column_ids'] = bos_column_ids + item_dict['input_column_ids'] + eos_column_ids
        item_dict['output_row_ids'] = bos_row_ids + item_dict['output_row_ids'] + eos_row_ids
        item_dict['output_column_ids'] = bos_column_ids + item_dict['output_column_ids'] + eos_column_ids

    def _pad_and_trunc(self, id_list: List, padding_id: int, trunc_eos_id: int, max_length: int):
        if len(id_list) <= max_length:  # pad
            return id_list + [padding_id] * (max_length - len(id_list))
        else:
            return id_list[:max_length - 1] + [trunc_eos_id]

    def _build_attention_mask(self, item_dict: Dict):
        if self.attention_mask_type == 'sequence':
            return
        elif self.attention_mask_type == 'row_column':
            input_ids, attention_mask = item_dict['input_ids'], item_dict['attention_mask']
            input_row_ids, input_column_ids = item_dict['input_row_ids'], item_dict['input_column_ids']
            seq_len = len(input_ids)
            new_attention_mask = attention_mask.repeat(seq_len, 1)
            row_mask = input_row_ids.eq(input_row_ids.view(seq_len, 1))
            column_mask = input_column_ids.eq(input_column_ids.view(seq_len, 1))
            row_column_mask = row_mask | column_mask
            text_token_mask = (input_row_ids == self.max_row_size) & (attention_mask == 1)
            non_text_token_mask = (input_row_ids != self.max_row_size) & (attention_mask == 1)
            row_column_text_mask = text_token_mask.repeat(torch.sum(non_text_token_mask), 1)
            new_attention_mask[non_text_token_mask] = (
                        row_column_mask[non_text_token_mask] | row_column_text_mask).type(torch.long)
            item_dict['attention_mask'] = new_attention_mask
        else:
            raise ValueError(f"{self.attention_mask_type} is currently not supported.")
