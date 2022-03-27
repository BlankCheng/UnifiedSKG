import unittest
from icecream import ic
from transformers import AutoTokenizer

from utils.processor.table_linearize import PretrainWikiTableLinearize


class TestPretrainLinearize(unittest.TestCase):

    def test_process_table(self):
        question = "what's the gdp of china?"
        header = ['', 'china', 'u.s.', 'japan']
        rows = [['gdp', '100', '200', '300'],
                ['population', '13', '', '1']]
        table_content = {
            "header": header,
            "rows": rows
        }
        for model_name in ['facebook/bart-base', 't5-base']:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            linearize_func = PretrainWikiTableLinearize(
                tokenizer=tokenizer,
                row_sep='row :',
                column_sep='|',
            )
            self._test_linearize(tokenizer, question, table_content, linearize_func)

    def _test_linearize(self, tokenizer, question, table_content, linearize_func):
        tokens, row_ids, column_ids = linearize_func.process_table_text(question, table_content)
        print('*' * 80)
        ic(len(row_ids), len(column_ids), len(tokens))
        for idx, token in enumerate(tokens, 0):
            print(token, row_ids[idx], column_ids[idx])
        print(tokenizer.convert_tokens_to_string(tokens))

        self.assertEqual(len(row_ids), len(column_ids))
        self.assertEqual(len(row_ids), len(tokens))


if __name__ == '__main__':
    unittest.main()
