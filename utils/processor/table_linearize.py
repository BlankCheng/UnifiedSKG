# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Utils for linearizing the table content into a flatten sequence
"""
import abc
from typing import Dict, List
import re
from transformers import AutoTokenizer, BasicTokenizer, BartTokenizer, BartTokenizerFast


class TableLinearize(abc.ABC):
    PROMPT_MESSAGE = """
        Please check that your table must follow the following format:
        {"header": ["col1", "col2", "col3"], "rows": [["row11", "row12", "row13"], ["row21", "row22", "row23"]]}
    """

    def process_table(self, table_content: Dict) -> str:
        """
        Given a table, TableLinearize aims at converting it into a flatten sequence with special symbols.
        """
        pass

    def process_header(self, headers: List):
        """
        Given a list of headers, TableLinearize aims at converting it into a flatten sequence with special symbols.
        """
        pass

    def process_row(self, row: List, row_index: int):
        """
        Given a row, TableLinearize aims at converting it into a flatten sequence with special symbols.
        """
        pass


class IndexedRowTableLinearize(TableLinearize):
    """
    FORMAT: col: col1 | col2 | col 3 row 1 : val1 | val2 | val3 row 2 : ...
    """

    def process_table(self, table_content: Dict):
        """
        Given a table, TableLinearize aims at converting it into a flatten sequence with special symbols.
        """
        assert "header" in table_content and "rows" in table_content, self.PROMPT_MESSAGE
        # process header
        _table_str = self.process_header(table_content["header"]) + " "
        # process rows
        for i, row_example in enumerate(table_content["rows"]):
            # NOTE: the row should start from row 1 instead of 0
            _table_str += self.process_row(row_example, row_index=i + 1) + " "
        return _table_str.strip()

    def process_header(self, headers: List):
        """
        Given a list of headers, TableLinearize aims at converting it into a flatten sequence with special symbols.
        """
        return "col : " + " | ".join(headers)

    def process_row(self, row: List, row_index: int):
        """
        Given a row, TableLinearize aims at converting it into a flatten sequence with special symbols.
        """
        row_str = ""
        row_cell_values = []
        for cell_value in row:
            if isinstance(cell_value, int):
                row_cell_values.append(str(cell_value))
            else:
                row_cell_values.append(cell_value)
        row_str += " | ".join(row_cell_values)
        return "row " + str(row_index) + " : " + row_str


class PretrainWikiTableLinearizeDeprecated(abc.ABC):
    """
    FORMAT: col: col1 | col2 | col 3 | row 1 : val1 | val2 | val3 | row 2 : ...
    """

    def __init__(self, tokenizer: BasicTokenizer = None, row_sep: str = "row :", column_sep: str = "|",
                 max_row_size: int = 256, max_column_size: int = 256, empty_cell_token: str = '<emp>'):
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="facebook/bart-large")
        else:
            self.tokenizer = tokenizer
        self.row_sep = row_sep.strip()
        self.column_sep = column_sep.strip()
        self.max_row_size = max_row_size
        self.max_column_size = max_column_size
        self.empty_cell_token = empty_cell_token

        # TODO: currently `row_sep` or `column_sep` can't be '' or ' ',
        #  since bart will tokenize f" {self.row_sep}" as `G', while not so in the whole linear string.
        assert self.row_sep != '' and self.column_sep != '', \
            f"Empty sep may cause mis-alignment in the current `token->string->token` ugly loop."
        # TODO: currently `empty_cell_token` can't be ''
        assert self.empty_cell_token != '', \
            "Empty cell token '' will disorder linearization since tokenizer may tokenize continuous space tokens."

    def process_table(self, table_content: Dict):
        """
        Given a table, TableLinearize aims at converting it into a flatten sequence with special symbols.
        """
        assert "header" in table_content and "rows" in table_content
        token2cell_map, cell2token_map = {}, {}
        row_ids, column_ids = [], []
        # process header
        _table_str, current_token_idx = self.process_header(
            headers=table_content["header"],
            row_ids=row_ids,
            column_ids=column_ids,
            cell2token_map=cell2token_map,
            token2cell_map=token2cell_map
        )
        _table_str += ' '
        # process rows
        for i, row_example in enumerate(table_content["rows"]):
            # NOTE: the row should start from row 1 instead of 0
            row_str, current_token_idx = self.process_row(
                row=row_example,
                row_index=i + 1,
                start_token_idx=current_token_idx,
                row_ids=row_ids,
                column_ids=column_ids,
                cell2token_map=cell2token_map,
                token2cell_map=token2cell_map
            )
            _table_str += row_str
            # _table_str += ' '
        _table_str = re.sub('\s+', ' ', _table_str.strip())
        return _table_str, row_ids, column_ids, cell2token_map, token2cell_map

    def process_header(
            self,
            headers: List,
            row_ids: List,
            column_ids: List,
            cell2token_map: Dict,
            token2cell_map: Dict
    ):
        """
        Given a list of headers, TableLinearize aims at converting it into a flatten sequence with special symbols.
        """
        row_sep_tokens = self.tokenizer.tokenize(self.row_sep)
        current_token_idx = len(row_sep_tokens)  # `col`, `:`
        linear_header_str = self.row_sep.replace('row', 'col')
        row_ids.extend([0 for _ in range(current_token_idx)])
        column_ids.extend([self.max_column_size for _ in range(current_token_idx)])
        for col_idx, header in enumerate(headers):
            if header == '':  # handle empty cell
                header = self.empty_cell_token
            coord = (0, col_idx)

            linear_header_str, current_token_idx = self._update_linear_cell_str(
                linear_str=linear_header_str,
                current_token_idx=current_token_idx,
                coord=coord,
                cell_value=header,
                row_ids=row_ids,
                column_ids=column_ids,
                token2cell_map=token2cell_map,
                cell2token_map=cell2token_map
            )
        return linear_header_str, current_token_idx

    def process_row(
            self,
            row: List,
            row_index: int,
            start_token_idx: int,
            row_ids: List,
            column_ids: List,
            cell2token_map: Dict,
            token2cell_map: Dict
    ):
        """
        Given a row, TableLinearize aims at converting it into a flatten sequence with special symbols.
        """
        if self.row_sep.strip() == 'row :':
            linear_row_str = f" row {row_index} :"
        else:
            linear_row_str = f" {self.row_sep}"
        row_sep_tokens = self.tokenizer.tokenize(linear_row_str)
        current_token_idx = start_token_idx + len(row_sep_tokens)  # `row` `5` `:`
        row_ids.extend([0 for _ in range(current_token_idx - start_token_idx)])
        column_ids.extend([self.max_column_size for _ in range(current_token_idx - start_token_idx)])
        for col_idx, cell_value in enumerate(row):
            if cell_value == '':  # handle empty cell
                cell_value = self.empty_cell_token
            coord = (row_index, col_idx)
            cell_value = str(cell_value)

            linear_row_str, current_token_idx = self._update_linear_cell_str(
                linear_str=linear_row_str,
                current_token_idx=current_token_idx,
                coord=coord,
                cell_value=cell_value,
                row_ids=row_ids,
                column_ids=column_ids,
                token2cell_map=token2cell_map,
                cell2token_map=cell2token_map
            )
        return linear_row_str, current_token_idx

    def _update_linear_cell_str(
            self,
            linear_str,
            current_token_idx,
            coord,
            cell_value,
            row_ids,
            column_ids,
            token2cell_map,
            cell2token_map
    ):
        """
        A utility function to linearize a cell string and update corresponding records.
        """
        linear_str += f" {cell_value}"
        linear_str += f" {self.column_sep}"
        cell_tokens = self.tokenizer.tokenize(f" {cell_value}")
        column_sep_tokens = self.tokenizer.tokenize(f" {self.column_sep}")
        cell_row_ids = [coord[0] for _ in range(len(cell_tokens) + len(self.column_sep.split()))]
        cell_column_ids = [coord[1] for _ in range(len(cell_tokens) + len(self.column_sep.split()))]
        row_ids.extend(cell_row_ids)
        column_ids.extend(cell_column_ids)
        token2cell_map[current_token_idx] = {
            'end_idx': current_token_idx + len(cell_tokens) - 1,
            'coord': coord
        }
        cell2token_map[coord] = [current_token_idx, current_token_idx + len(cell_tokens) - 1]
        current_token_idx += len(cell_tokens)
        current_token_idx += len(column_sep_tokens)

        return linear_str, current_token_idx


class PretrainWikiTableLinearize(abc.ABC):
    """
    FORMAT: col: col1 | col2 | col 3 | row 1 : val1 | val2 | val3 | row 2 : ...
    """

    def __init__(
            self,
            tokenizer: BasicTokenizer = None,
            row_sep: str = "row :",
            column_sep: str = "|",
            max_row_size: int = 256,
            max_column_size: int = 256,
            sep_token: str = "  ",
            nl_template: str = None  # TODO: use template to linearize table as NL string
    ):
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="facebook/bart-large")
        else:
            self.tokenizer = tokenizer
        self.row_sep = row_sep.strip()
        self.column_sep = column_sep.strip()
        if isinstance(self.tokenizer, BartTokenizer) \
                or isinstance(self.tokenizer, BartTokenizerFast):
            if self.row_sep != '':
                self.row_sep = " " + self.row_sep
            if self.column_sep != '':
                self.column_sep = " " + self.column_sep
        self.max_row_size = max_row_size
        self.max_column_size = max_column_size
        self.sep_token = sep_token

    def process_table_text(self, text: str, table_content: Dict):
        text_tokens, text_row_ids, text_column_ids = self.process_text(text)
        table_tokens, table_row_ids, table_column_ids = self.process_table(table_content)
        sep_tokens = self.tokenizer.tokenize(self.sep_token)
        tokens = text_tokens + sep_tokens + table_tokens
        row_ids = text_row_ids + [self.max_row_size] * len(sep_tokens) + table_row_ids
        column_ids = text_column_ids + [self.max_column_size] * len(sep_tokens) + table_column_ids
        return tokens, row_ids, column_ids

    def process_text(self, text: str):
        tokens = self.tokenizer.tokenize(text)
        row_ids = [self.max_row_size for _ in tokens]
        column_ids = [self.max_column_size for _ in tokens]
        return tokens, row_ids, column_ids

    def process_table(self, table_content: Dict):
        """
        Given a table, TableLinearize aims at converting it into a flatten sequence with special symbols.
        """
        assert "header" in table_content and "rows" in table_content
        # process header
        tokens, row_ids, column_ids = [], [], []
        self.process_header(
            headers=table_content["header"],
            tokens=tokens,
            row_ids=row_ids,
            column_ids=column_ids,
        )
        # process rows
        for i, row_example in enumerate(table_content["rows"]):
            # NOTE: the row should start from row 1 instead of 0
            self.process_row(
                row=row_example,
                row_index=i + 1,
                tokens=tokens,
                row_ids=row_ids,
                column_ids=column_ids,
            )
        return tokens, row_ids, column_ids

    def process_header(
            self,
            headers: List,
            tokens: List,
            row_ids: List,
            column_ids: List,
    ):
        # tokenize row separator
        row_sep = self.row_sep.replace("row", 'col')
        row_sep_tokens = self.tokenizer.tokenize(row_sep)
        tokens.extend(row_sep_tokens)
        row_ids.extend([0 for _ in row_sep_tokens])
        column_ids.extend([self.max_column_size for _ in row_sep_tokens])
        # tokenize header row
        column_sep_tokens = self.tokenizer.tokenize(self.column_sep)
        for column_idx, header in enumerate(headers):
            header = str(header).strip()
            if isinstance(self.tokenizer, BartTokenizer) \
                    or isinstance(self.tokenizer, BartTokenizerFast):  # To keep the same with downstream tasks
                header = " " + header
            cell_tokens = self.tokenizer.tokenize(header)
            tokens.extend(cell_tokens + column_sep_tokens)
            row_ids.extend([0 for _ in cell_tokens + column_sep_tokens])
            column_ids.extend([column_idx for _ in cell_tokens + column_sep_tokens])

    def process_row(
            self,
            row: List,
            row_index: int,
            tokens: List,
            row_ids: List,
            column_ids: List
    ):
        # tokenize row separator
        row_sep = self.row_sep
        if self.row_sep.strip() == 'row :':
            row_sep = f"row {row_index} :"
        if self.row_sep.startswith(" "):
            row_sep = " " + row_sep
        row_sep_tokens = self.tokenizer.tokenize(row_sep)
        tokens.extend(row_sep_tokens)
        row_ids.extend([row_index for _ in row_sep_tokens])
        column_ids.extend([self.max_column_size for _ in row_sep_tokens])
        # tokenize content row
        column_sep_tokens = self.tokenizer.tokenize(self.column_sep)
        for column_idx, cell_value in enumerate(row):
            cell_value = str(cell_value).strip()
            if isinstance(self.tokenizer, BartTokenizer) \
                    or isinstance(self.tokenizer, BartTokenizerFast):  # To keep the same with downstream tasks
                cell_value = " " + cell_value
            cell_tokens = self.tokenizer.tokenize(cell_value)
            tokens.extend(cell_tokens + column_sep_tokens)
            row_ids.extend([row_index for _ in cell_tokens + column_sep_tokens])
            column_ids.extend([column_idx for _ in cell_tokens + column_sep_tokens])