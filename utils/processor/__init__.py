"""
Processor adapts from TAPEX code, see https://github.com/microsoft/Table-Pretraining
(We need to add more features in table handling)
"""
from .table_linearize import IndexedRowTableLinearize, PretrainWikiTableLinearize
from .table_processor import TableProcessor, PretrainWikiTableProcessor
from .table_truncate import CellLimitTruncate, RowDeleteTruncate
from .pretrain_tool import PretrainTool, PretrainMLMTool, PretrainToTensorTool


def get_default_processor(tokenizer, max_cell_length, max_input_length):
    table_linearize_func = IndexedRowTableLinearize()
    table_truncate_funcs = [
        CellLimitTruncate(max_cell_length=max_cell_length,
                          tokenizer=tokenizer,
                          max_input_length=max_input_length),
        RowDeleteTruncate(table_linearize=table_linearize_func,
                          tokenizer=tokenizer,
                          max_input_length=max_input_length)
    ]
    processor = TableProcessor(table_linearize_func=table_linearize_func,
                               table_truncate_funcs=table_truncate_funcs)
    return processor


# Pretrain
PRETRAIN_PROCESS_TOOL_DICT = {
    'mlm': PretrainMLMTool
}


def get_pretrain_wikitable_processor(
        args,
        tokenizer,
        model_config
):
    table_linearize_func = PretrainWikiTableLinearize(
        tokenizer=tokenizer,
        row_sep=args.preprocess.row_sep,
        column_sep=args.preprocess.column_sep
    )
    table_truncate_funcs = [
        CellLimitTruncate(max_cell_length=args.preprocess.max_cell_length,
                          tokenizer=tokenizer,
                          max_input_length=args.preprocess.max_table_truncation_length),
        RowDeleteTruncate(table_linearize=IndexedRowTableLinearize(),
                          tokenizer=tokenizer,
                          max_input_length=args.preprocess.max_table_truncation_length)
    ]

    assert args.preprocess.pretrain_task in PRETRAIN_PROCESS_TOOL_DICT.keys()

    pretrain_process_tool = PRETRAIN_PROCESS_TOOL_DICT[args.preprocess.pretrain_task](
        tokenizer=tokenizer,
        mlm_rate=args.preprocess.mlm_rate,
        table_mask_type=args.preprocess.table_mask_type,
        row_sep=args.preprocess.row_sep,
        column_sep=args.preprocess.column_sep,
        max_row_size=args.preprocess.max_row_size,
        max_column_size=args.preprocess.max_column_size
    )

    pretrain_totensor_tool = PretrainToTensorTool(
        tokenizer=tokenizer,
        model_config=model_config,
        max_input_length=args.preprocess.max_input_length,
        max_generation_length=args.preprocess.max_generation_length,
        max_row_size=args.preprocess.max_row_size,
        max_column_size=args.preprocess.max_column_size,
        attention_mask_type=args.preprocess.attention_mask_type
    )

    processor = PretrainWikiTableProcessor(table_linearize_func=table_linearize_func,
                                           table_truncate_funcs=table_truncate_funcs,
                                           pretrain_process_tool=pretrain_process_tool,
                                           pretrain_totensor_tool=pretrain_totensor_tool)

    return processor
