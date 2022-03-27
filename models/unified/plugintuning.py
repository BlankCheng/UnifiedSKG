#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch import nn
from .base import PushToHubFriendlyModel
from transformers import (
    AutoTokenizer,
    AutoConfig
)
from ..plugin.modeling_auto import AutoModelForSeq2SeqLM


PLUGIN_PARAM_KEYS = [  # TODO: should be changed when model arch changes
    'row_weights',
    'column_weights',
    'embed_rows',
    'embed_columns',
    'plugin',
    'top_attn_layer',
    'top_fc_layer'
]


class Model(PushToHubFriendlyModel):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # Load tokenizer and model.
        self.tokenizer = AutoTokenizer.from_pretrained(args.bert.location, use_fast=False)
        self.config = AutoConfig.from_pretrained(args.bert.location)
        args_dict = self.args.to_dict()
        self.config.update(args_dict)
        self.pretrain_model = AutoModelForSeq2SeqLM.from_config(
            self.config
        )

        if args.special_tokens:
            self.tokenizer.add_tokens([v for k, v in args.special_tokens])
            self.pretrain_model.resize_token_embeddings(len(self.tokenizer))

        if self.config.model['freeze_plm']:
            for name, param in self.pretrain_model.named_parameters():
                freeze_flag = True
                for plugin_key in PLUGIN_PARAM_KEYS:
                    if plugin_key in name:
                        freeze_flag = False
                        break
                if freeze_flag:
                    param.requires_grad = False
                else:
                    print(f"{name} -> will update with gradients.")

    def forward(
            self,
            input_ids,
            attention_mask,
            labels,
            decoder_input_ids=None,
            row_ids=None,
            column_ids=None,
            **kwargs
    ):
        loss = self.pretrain_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            use_cache=False,
            labels=labels,
            row_ids=row_ids,
            column_ids=column_ids
        ).loss
        return {'loss': loss}

    def generate(self, input_ids, attention_mask, **kwargs):
        generated_ids = self.pretrain_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            **kwargs,
        )

        return generated_ids
