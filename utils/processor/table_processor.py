# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Dict, List
from .table_linearize import TableLinearize, PretrainWikiTableLinearize
from .table_truncate import TableTruncate
from .pretrain_tool import PretrainMLMTool, PretrainToTensorTool


class TableProcessor(object):

    def __init__(self, table_linearize_func: TableLinearize,
                 table_truncate_funcs: List[TableTruncate],
                 target_delimiter: str = ", "):
        self.table_linearize_func = table_linearize_func
        self.table_truncate_funcs = table_truncate_funcs
        self.target_delimiter = target_delimiter

    def process_input(self, table_content: Dict, question: str, answer: List[str]) -> str:
        """
        Preprocess a sentence into the expected format for model translate.
        """
        # modify a table internally
        for truncate_func in self.table_truncate_funcs:
            truncate_func.truncate_table(table_content, question, answer)
        # linearize a table into a string
        linear_table = self.table_linearize_func.process_table(table_content)
        # concat question with linear_table
        joint_input = question + " " + linear_table
        return joint_input

    def process_output(self, answer: List[str]) -> str:
        """
        Flatten the output for translation
        """
        output = self.target_delimiter.join(answer)
        if output.strip() == "":
            raise Exception("The Answer is EMPTY!")
        else:
            return output


class PretrainWikiTableProcessor(object):
    _MAX_PRETRAIN_QUESTION_WORDS = 32

    def __init__(self, table_linearize_func: PretrainWikiTableLinearize,
                 table_truncate_funcs: List[TableTruncate],
                 pretrain_process_tool: PretrainMLMTool,
                 pretrain_totensor_tool: PretrainToTensorTool,
                 target_delimiter: str = ", "):
        self.table_linearize_func = table_linearize_func
        self.table_truncate_funcs = table_truncate_funcs
        self.pretrain_process_tool = pretrain_process_tool
        self.pretrain_totensor_tool = pretrain_totensor_tool
        self.target_delimiter = target_delimiter

    def process(self, table_content: Dict[str, List], question: str):
        """
        Given the pretrain question and table, process them into pretrain task input and output.
        """
        # normalize
        table_content, question = self._normalize_table_question(table_content, question)
        # modify a table internally
        for truncate_func in self.table_truncate_funcs:
            truncate_func.truncate_table(table_content=table_content, question=question, answer=[])
        # linearize the table and text into tokens, with row and column ids
        tokens, row_ids, column_ids = self.table_linearize_func.process_table_text(
            text=question,
            table_content=table_content
        )
        # process to get pretrain task-specific joint input and output
        item_dict = self.pretrain_process_tool.pretrain_process(
            tokens=tokens,
            row_ids=row_ids,
            column_ids=column_ids
        )
        # build attention mask and to tensor
        item_tensor_dict = self.pretrain_totensor_tool.pretrain_to_tensor(item_dict)

        return item_tensor_dict

    def _normalize_table_question(self, table_content, question):
        for idx, header in enumerate(table_content['header']):
            table_content['header'][idx] = self._normalize_str(header)
        for row in table_content['rows']:
            for idx, cell in enumerate(row):
                row[idx] = self._normalize_str(cell)
        question = ' '.join(question.split()[:PretrainWikiTableProcessor._MAX_PRETRAIN_QUESTION_WORDS])
        question = self._normalize_str(question)
        return table_content, question

    def _normalize_str(self, string):
        # TODO: add other normalizations
        return string.replace('\n', ' ').strip()

