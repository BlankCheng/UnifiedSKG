import unittest
from icecream import ic
from transformers import AutoTokenizer, AutoConfig

from utils.processor.table_linearize import PretrainWikiTableLinearize
from utils.processor.pretrain_tool import PretrainMLMTool, PretrainToTensorTool


class TestPretrainTool(unittest.TestCase):
    def test_pretrain_tool(self):
        question = 'title : Nikto, krome nas... ; description : Nikto, krome nas... is a 2008 ' \
                    'Russian war film directed by Sergey Govorukhin. Plot Overview: The Russian ' \
                    'military participated directly in war with Tajikistan'
        header = ['', 'china', 'u.s.\n', 'japan island']
        rows = [['gdp', '100 Fer 1930', '200 hello ', '300'],
                ['population land', '13 nera', '', '1'],
                ['Replay', 'TUSD', '18 February', 'Something']]
        table_content = {
            "header": header,
            "rows": rows
        }
        for model_name in ['facebook/bart-base', 't5-base']:
            # init
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model_config = AutoConfig.from_pretrained(model_name)
            linearize_func = PretrainWikiTableLinearize(
                tokenizer=tokenizer,
                row_sep='row :',
                column_sep='|',
            )
            pretrain_process_tool = PretrainMLMTool(
                tokenizer=tokenizer,
                mlm_rate=0.4,
                table_mask_type='cell',
            )
            pretrain_totensor_tool = PretrainToTensorTool(
                tokenizer=tokenizer,
                model_config=model_config,
                max_input_length=64,
                max_generation_length=64,
                attention_mask_type='sequence'
            )

            # linearize
            tokens, row_ids, column_ids = linearize_func.process_table_text(question, table_content)

            # pretrain mlm processing
            item_dict = pretrain_process_tool.pretrain_process(tokens, row_ids, column_ids)

            # pretrain mlm precessing (debug)
            input_tokens, input_row_ids, input_column_ids = \
                item_dict['input_tokens'], item_dict['input_row_ids'], item_dict['input_column_ids']
            output_tokens, output_row_ids, output_column_ids = \
                item_dict['output_tokens'], item_dict['output_row_ids'], item_dict['output_column_ids']
            # print('*' * 80)
            # ic(len(input_tokens), len(input_row_ids), len(input_column_ids))
            # for idx, input_token in enumerate(input_tokens, 0):
            #     print(input_token, input_row_ids[idx], input_column_ids[idx])
            # print(tokenizer.convert_tokens_to_string(input_tokens))
            # print('-' * 80)
            # ic(len(output_tokens), len(output_row_ids), len(output_column_ids))
            # for idx, output_token in enumerate(output_tokens, 0):
            #     print(output_token, output_row_ids[idx], output_column_ids[idx])
            # print(tokenizer.convert_tokens_to_string(output_tokens))
            self.assertEqual(len(input_row_ids), len(input_column_ids))
            self.assertEqual(len(input_row_ids), len(input_tokens))
            self.assertEqual(len(output_row_ids), len(output_column_ids))
            self.assertEqual(len(output_row_ids), len(output_tokens))

            # pretrain to tensor
            item_tensor_dict = pretrain_totensor_tool.pretrain_to_tensor(item_dict)

            # pretrain to tensor (debug)
            input_ids, input_tokens = item_tensor_dict['input_ids'], item_dict['input_tokens']
            attention_mask = item_tensor_dict['attention_mask']
            labels, output_tokens = item_tensor_dict['labels'], item_dict['output_tokens']
            input_row_ids, input_column_ids = item_tensor_dict['input_row_ids'], item_tensor_dict['input_column_ids']
            output_row_ids, output_column_ids = item_tensor_dict['output_row_ids'], item_tensor_dict['output_column_ids']
            for idx, input_id in enumerate(input_ids, 0):
                input_token = tokenizer.convert_ids_to_tokens([input_id])
                print(input_id, input_token, input_row_ids[idx], input_column_ids[idx], "attn_mask:", attention_mask[idx])
            print(tokenizer.convert_tokens_to_string(input_tokens))
            print('-' * 80)
            for idx, output_id in enumerate(labels, 0):
                if output_id == -100:
                    output_token = '<p>'
                else:
                    output_token = tokenizer.convert_ids_to_tokens([output_id])
                print(output_id, output_token, output_row_ids[idx], output_column_ids[idx])
            print(tokenizer.convert_tokens_to_string(output_tokens))
            self.assertEqual(len(input_ids), len(attention_mask))
            self.assertEqual(len(input_ids), len(input_row_ids))
            self.assertEqual(len(input_row_ids), len(input_column_ids))
            self.assertEqual(len(labels), len(output_row_ids))
            self.assertEqual(len(output_row_ids), len(output_column_ids))


if __name__ == '__main__':
    unittest.main()
