#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch import nn
import torch
from .base import PushToHubFriendlyModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class Model(PushToHubFriendlyModel):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # Load tokenizer and model.
        self.tokenizer = AutoTokenizer.from_pretrained(args.bert.location, use_fast=False)
        self.pretrain_model = AutoModelForSeq2SeqLM.from_pretrained(
            args.bert.location,
        )
        self.config = self.pretrain_model.config

        if args.special_tokens:
            self.tokenizer.add_tokens([v for k, v in args.special_tokens])
            self.pretrain_model.resize_token_embeddings(len(self.tokenizer))

        if args.bert.extra_pretrain_path:
            self.extra_init_pretrain_model(args.bert.extra_pretrain_path)

    def forward(self, input_ids, attention_mask, labels, **kwargs):
        loss = self.pretrain_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            labels=labels,
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

    def extra_init_pretrain_model(self, extra_pretrain_path, strict=True):
        """
        Load extra pretrained model form local path.
        """
        pretrain_dict = torch.load(extra_pretrain_path,
                                   map_location=torch.device('cpu'))

        self.load_state_dict(pretrain_dict, strict=strict)
        print(f"Load {len(pretrain_dict)} params(trict={strict}) to pretrained_model.")

