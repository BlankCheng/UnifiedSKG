""" A python class alternative for 'interaction.proto' in Tapas. """

from typing import Dict, List, Union, Any


class Date:
    """ Simple Date class."""
    def __init__(self, year: int, month: int, day: int):
        self.year = year
        self.month = month
        self.day = day

    def __str__(self):
        return f"Date(year={self.year}), month={self.month}, day={self.day}"


class Cell:
    """ Table cell class."""
    def __init__(self, text: str, numeric_value: Union[float, Date]):
        self.text = text
        self.numeric_value = numeric_value

    def __str__(self):
        if self.numeric_value:
            return f"{self.text}|num={str(self.numeric_value)}"
        else:
            return f"{self.text}"


class Table:
    """ Table class."""
    def __init__(self,
                 columns: List[Cell],
                 rows: List[List[Cell]],
                 table_id: str,
                 document_title: str,
                 caption: str,
                 document_url: str,
                 alternative_document_urls: List[str],
                 alternative_table_ids: List[str],
                 context_heading: str
                 ):
        self.columns = columns
        self.rows = rows
        self.table_id = table_id
        self.document_title = document_title
        self.caption = caption
        self.document_url = document_url
        self.alternative_document_urls = alternative_document_urls
        self.alternative_table_ids = alternative_table_ids
        self.context_heading = context_heading

    def __str__(self):
        str_columns = str([str(cell) for cell in self.columns])
        str_cells = [[str(cell) for cell in row] for row in self.rows]
        return f"Table\n" \
               f"\tColumns: {str_columns}\n" \
               f"\tCells: {str_cells}\n"


class Question:
    """ Question class."""
    def __init__(self, id: str, text: str, original_text: str):
        self.id = id
        self.text = text
        self.original_text = original_text

    def __str__(self):
        return f"Question\n" \
               f"\t{self.id}-> {self.original_text}\n"


class WikiTableSample:
    """ WikiTable sample class for Tapas pretrain corpus."""
    def __init__(self, id: str, table: Table, questions: Dict[str, Question]):
        self.id = id
        self.table = table
        self.questions = questions

    def __str__(self):
        str_sample = f"Sample({self.id})\n"
        str_sample += str(self.table)
        for _, question in self.questions.items():
            str_sample += str(question)
        return str_sample


