# -*- coding: utf-8 -*-
"""
Backbones of Pre-training Models (from input to last hidden-layer output)
"""

from more_itertools import last
import torch
import torch.nn as nn
from torch.nn.modules import dropout
import tuta.model.embeddings as emb
import tuta.model.encoders as enc
from transformers import BertConfig  # noqa
from icecream import ic
from timm.models.layers import trunc_normal_
from torch.autograd import variable


class Backbone(nn.Module):
    def __init__(self, config):
        super(Backbone, self).__init__()
        self.total_node = sum(config.node_degree)
        self.attn_method = config.attn_method
        self.attn_methods = {"max": self.pos2attn_max,
                             "add": self.pos2attn_add}

    def unzip_tree_position(self, zipped_position):
        """
        args: zipped_position: [batch_size, seq_len, tree_depth], range: [0, total_node]
        rets: entire_position: [batch_size, seq_len, total_node]
        lower_bound = 0, upper_bound = (total_node-1)
        use one excessive bit to temporarily represent not-applicable nodes
        """
        batch_size, seq_len, _ = zipped_position.size()
        entire_position = torch.zeros(batch_size, seq_len, self.total_node + 1).to(zipped_position.device)
        entire_position = entire_position.scatter_(-1, zipped_position, 1.0).long()
        entire_position = entire_position[:, :, : self.total_node]  # remove last column
        return entire_position

    def get_attention_mask(self, entire_top, entire_left, indicator):
        # attention_mask = self.attn_methods[self.attn_method](entire_top, entire_left)  # TODO
        b, seq_len = indicator.size()
        attention_mask = torch.zeros((b, seq_len, seq_len)).to(indicator.device)
        attention_mask = self.create_post_mask(attention_mask, indicator)
        return attention_mask

    def pos2attn_max(self, pos_top, pos_left):  # entire position
        top_attn_mask = self.pos2attn(pos_top)
        left_attn_mask = self.pos2attn(pos_left)
        attn_mask = torch.max(top_attn_mask, left_attn_mask)
        # attn_mask = top_attn_mask + left_attn_mask
        return attn_mask

    def pos2attn_add(self, pos_top, pos_left):  # entire position
        top_attn_mask = self.pos2attn(pos_top)
        left_attn_mask = self.pos2attn(pos_left)
        attn_mask = top_attn_mask + left_attn_mask
        return attn_mask

    def pos2attn(self, position):  # entire position # FIXME: 占显存，pre-train的时候可以先注释掉，attn_mask置0或1
        """Compute a one-dimension attention distance matrix from a entire-mode tree position. """
        vector_matrix = position.unsqueeze(2).repeat(1, 1, position.size()[1],
                                                     1)  # [batch, seq_len, seq_len, total_node]
        attention_mask = torch.abs(vector_matrix - vector_matrix.transpose(1, 2))
        attention_mask = torch.sum(attention_mask, dim=-1)
        return attention_mask

    def create_post_mask(self, attn_dist, indicator, padding_dist=100):
        """
        [CLS] sees all of the tokens except for the [PAD]s
        [SEP]s in table see each other & their own cells; [SEP]s in clc/tcr choices see as their tokens
        Tokens see their friend and corresponding [SEP]
        """
        # cls_matrix = (indicator == -1).long().unsqueeze(-1).repeat(1, 1, attn_dist.size(1))
        # cls_matrix = torch.max(cls_matrix, cls_matrix.transpose(-1, -2))
        # cls_matrix = -(cls_matrix * attn_dist)
        # pad_matrix = (indicator == 0).long().unsqueeze(-1).repeat(1, 1, attn_dist.size(1))
        # pad_matrix = torch.max(pad_matrix, pad_matrix.transpose(-1, -2)) * padding_dist
        # attn_dist = attn_dist + cls_matrix + pad_matrix
        #
        # # only table-[SEP]s and root can see their contexts
        # sep_matrix = (indicator > 0).long() * (indicator%2 == 1).long()
        # sep_matrix = sep_matrix.unsqueeze(-1).repeat(1, 1, attn_dist.size(1))
        # sep_matrix = (1 - sep_matrix * sep_matrix.transpose(1, 2)) * padding_dist
        # attn_dist = attn_dist * (sep_matrix + 1)

        # new added
        pad_matrix = (indicator == 0).long().unsqueeze(-1).repeat(1, 1, attn_dist.size(1))
        pad_matrix = torch.max(pad_matrix, pad_matrix.transpose(-1, -2)) * padding_dist
        attn_dist = attn_dist + pad_matrix
        return attn_dist


class BbForBase(Backbone):
    def __init__(self, config):
        super(Backbone, self).__init__()
        self.embeddings = emb.EmbeddingForBase(config)
        self.encoder = enc.Encoder(config)
        self.attn_methods = {"max": self.pos2attn_max,
                             "add": self.pos2attn_add}
        self.attn_method = config.attn_method
        self.total_node = sum(config.node_degree)

    def forward(self, token_id, num_mag, num_pre, num_top, num_low, token_order, pos_top, pos_left, format_vec,
                indicator):
        embedded_states = self.embeddings(token_id, num_mag, num_pre, num_top, num_low, token_order, format_vec)
        entire_pos_top = self.unzip_tree_position(pos_top)
        entire_pos_left = self.unzip_tree_position(pos_left)
        attn_mask = self.get_attention_mask(entire_pos_top, entire_pos_left, indicator)
        encoded_states = self.encoder(embedded_states, attn_mask)
        return encoded_states


class BbForTutaExplicit(Backbone):
    def __init__(self, config):
        super(Backbone, self).__init__()
        self.embeddings = emb.EmbeddingForTutaExplicit(config)
        self.encoder = enc.Encoder(config)
        self.attn_methods = {"max": self.pos2attn_max,
                             "add": self.pos2attn_add}
        self.attn_method = config.attn_method
        self.total_node = sum(config.node_degree)

    def forward(self,
                token_id, num_mag, num_pre, num_top, num_low,
                token_order, pos_row, pos_col, pos_top, pos_left,
                format_vec, indicator
                ):
        entire_pos_top = self.unzip_tree_position(pos_top)
        entire_pos_left = self.unzip_tree_position(pos_left)
        embedded_states = self.embeddings(
            token_id, num_mag, num_pre, num_top, num_low,
            token_order, pos_row, pos_col, entire_pos_top, entire_pos_left, format_vec
        )
        attn_mask = self.get_attention_mask(entire_pos_top, entire_pos_left, indicator)
        encoded_states = self.encoder(embedded_states, attn_mask)
        return encoded_states


class BbForTuta(Backbone):
    def __init__(self, config):
        super(Backbone, self).__init__()
        self.embeddings = emb.EmbeddingForTuta(config)
        self.encoder = enc.Encoder(config)
        self.attn_methods = {"max": self.pos2attn_max,
                             "add": self.pos2attn_add}
        self.attn_method = config.attn_method
        self.total_node = sum(config.node_degree)

        self.config = BertConfig(
            vocab_size_or_config_json_file=config.vocab_size,
            attention_probs_dropout_prob=0.1,
            hidden_act=config.hidden_act,
            hidden_dropout_prob=config.hidden_dropout_prob,
            hidden_size=config.hidden_size,
            initializer_range=0.02,
            intermediate_size=config.intermediate_size,
            # layer_norm_eps=1e-12,
            max_position_embeddings=512,
            num_attention_heads=config.num_attention_heads,
            num_hidden_layers=12,
            type_vocab_size=2,
        )

    def forward(self,
                token_id, num_mag, num_pre, num_top, num_low,
                token_order, pos_row, pos_col, pos_top, pos_left,
                format_vec, indicator, row_index=None, column_index=None
                ):
        embedded_states = self.embeddings(
            token_id, num_mag, num_pre, num_top, num_low,
            token_order, pos_row, pos_col, pos_top, pos_left, format_vec
        )
        entire_pos_top = self.unzip_tree_position(pos_top)
        entire_pos_left = self.unzip_tree_position(pos_left)
        attn_mask = self.get_attention_mask(entire_pos_top, entire_pos_left, indicator)
        encoded_states = self.encoder(embedded_states, attn_mask)
        return encoded_states


# huggingface model
from transformers.models.bart.modeling_bart import *


# Inherit BartModel
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to()

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(dtype).min)


def _make_causal_mask(input_ids_shape: torch.Size, dtype: torch.dtype, past_key_values_length: int = 0):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), float("-inf"))
    mask_cond = torch.arange(mask.size(-1))
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


class BbBartAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            dropout: float = 0.0,
            is_decoder: bool = False,
            bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert (
                self.head_dim * num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {num_heads})."
        self.scaling = self.head_dim ** -0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # add bias:
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(4, self.num_heads)
        )
        self.parameter_table_index = {
            "other": 0,
            "same row": 1,
            "same col": 2,
            "same cell": 3,
        }
        trunc_normal_(self.relative_position_bias_table, std=.02)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def generate_bias(self, pos_row, pos_col):
        bsz, seq_len = pos_row.size()
        device = self.relative_position_bias_table.device
        indicator_row = pos_row.detach()
        coords_row = pos_row.unsqueeze(1).detach()
        coords_col = pos_col.unsqueeze(1).detach()
        coords_flatten = torch.cat([coords_row, coords_col], dim=1)

        relative_coords = coords_flatten[:, :, :, None] - coords_flatten[:, :, None, :]
        relative_row = relative_coords[:, 0, :]
        relative_col = relative_coords[:, 1, :]
        relative_cell = relative_row.abs() + relative_col.abs()
        relative_position_index = torch.zeros_like(relative_row)

        relative_position_index = torch.where(
            relative_row == 0,
            torch.ones_like(relative_row) * self.parameter_table_index["same row"],
            torch.zeros_like(relative_row)
        )
        relative_position_index = torch.where(
            relative_col == 0,
            torch.ones_like(relative_col) * self.parameter_table_index["same col"],
            relative_position_index
        )
        relative_position_index = torch.where(
            relative_cell == 0,
            torch.ones_like(relative_cell) * self.parameter_table_index["same cell"],
            relative_position_index
        )
        relative_position_index = relative_position_index.detach()
        relative_position_bias = self.relative_position_bias_table[relative_position_index.view(-1)].view(
            bsz, seq_len, seq_len, -1
        )
        # relative_position_bias[indicator_row == 256, :, :] *= 0
        # relative_position_bias = relative_position_bias

        return relative_position_bias

    def forward(
            self,
            hidden_states: torch.Tensor,
            key_value_states: Optional[torch.Tensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            attention_mask: Optional[torch.Tensor] = None,
            layer_head_mask: Optional[torch.Tensor] = None,
            output_attentions: bool = False,
            pos_row=None,
            pos_col=None,
            bias=False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, embed_dim = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        # before softmax, add bias
        # if bias and pos_row is not None and pos_col is not None:
        #     attn_relative_bias = self.generate_bias(pos_row, pos_col)
        #     attn_weights = attn_weights + attn_relative_bias.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value


class Plugin(nn.Module):
    def __init__(self,
                 embed_dim,
                 num_heads,
                 attn_dropout,
                 fn_dropout,
                 act_dropout,
                 outer_dim
                 ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ffn_dim = embed_dim
        self.attn_dropout = attn_dropout
        self.outer_dim = outer_dim
        self.self_attn = BbBartAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout=self.attn_dropout
        )
        self.self_attention_layer_norm = nn.LayerNorm(self.ffn_dim)
        self.dropout = nn.Dropout(p=fn_dropout)
        self.activation_fn = ACT2FN["gelu"]
        self.activation_dropout = nn.Dropout(p=act_dropout)
        self.fc1 = nn.Linear(self.outer_dim, self.ffn_dim)
        self.fc2 = nn.Linear(self.ffn_dim, self.outer_dim)
        self.final_layer_norm = nn.LayerNorm(self.outer_dim)

    def forward(
            self,
            hidden_pos_states: torch.Tensor,
            attention_mask: torch.Tensor,
            layer_head_mask: torch.Tensor,
            output_attentions: bool = False,
    ):
        # scale down, hidden_pos_states: (b, seq_len, out_dim) => (b, seq_len, embed_dim)
        hidden_pos_states = self.activation_fn(self.fc1(hidden_pos_states))
        hidden_pos_states = self.activation_dropout(hidden_pos_states)

        residual = hidden_pos_states
        hidden_pos_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_pos_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions
        )
        hidden_pos_states = self.dropout(hidden_pos_states)
        hidden_pos_states = residual + hidden_pos_states
        hidden_pos_states = self.self_attention_layer_norm(hidden_pos_states)

        # scale up, hidden_pos_states: (b, seq_len, embed_dim) => (b, seq_len, out_dim)
        hidden_pos_states = self.fc2(hidden_pos_states)
        hidden_pos_states = self.dropout(hidden_pos_states)
        hidden_pos_states = self.final_layer_norm(hidden_pos_states)

        if hidden_pos_states.dtype == torch.float16 and (
                torch.isinf(hidden_pos_states).any() or torch.isnan(hidden_pos_states).any()
        ):
            clamp_value = torch.finfo(hidden_pos_states.dtype).max - 1000
            hidden_pos_states = torch.clamp(hidden_pos_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_pos_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class PlugLayer(nn.Module):
    def __init__(self, config: BartConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = BbBartAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor,
            layer_head_mask: torch.Tensor,
            output_attentions: bool = False,
            pos_row=None,
            pos_col=None,
            hidden_position_states: torch.Tensor = None
    ):
        """
        Args:
            hidden_states (:obj:`torch.FloatTensor`): input to the layer of shape `(seq_len, batch, embed_dim)`
            attention_mask (:obj:`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (:obj:`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
        """
        # bias config
        bias = pos_row is not None and pos_col is not None
        residual = hidden_states
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
            pos_row=pos_row,
            pos_col=pos_col,
            bias=bias
        )
        # hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        # hidden_states = residual + hidden_states
        # hidden_states = self.self_attn_layer_norm(hidden_states)

        # residual = hidden_states
        # hidden_states = self.activation_fn(self.fc1(hidden_states))
        # hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        # hidden_states = self.fc2(hidden_states)
        # hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        # hidden_states = residual + hidden_states
        # hidden_states = self.final_layer_norm(hidden_states)

        # if hidden_states.dtype == torch.float16 and (
        #     torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        # ):
        #     clamp_value = torch.finfo(hidden_states.dtype).max - 1000
        #     hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class ExtendEncoderLayer(nn.Module):
    def __init__(self,
                 embed_dim,
                 num_heads,
                 attn_dropout,
                 fn_dropout,
                 act_dropout,
                 outer_dim
                 ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ffn_dim = embed_dim
        self.attn_dropout = attn_dropout
        self.outer_dim = outer_dim
        self.self_attn = BbBartAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout=self.attn_dropout
        )
        self.self_attention_layer_norm = nn.LayerNorm(self.ffn_dim)

        self.dropout = nn.Dropout(p=fn_dropout)
        self.activation_fn = ACT2FN["gelu"]
        self.activation_dropout = nn.Dropout(p=act_dropout)
        self.fc1 = nn.Linear(self.outer_dim, self.ffn_dim)
        self.fc2 = nn.Linear(self.ffn_dim, self.ffn_dim)
        self.final_layer_norm = nn.LayerNorm(self.ffn_dim)

        self.prev_fc1_weight = None
        self.cur_fc1_weight = None

    def forward(
            self,
            hidden_pos_states: torch.Tensor,
            attention_mask: torch.Tensor,
            layer_head_mask: torch.Tensor,
            output_attentions: bool = False,
            use_plugin: bool = True
    ):
        # scale down, hidden_pos_states: (b, seq_len, out_dim) => (b, seq_len, embed_dim)
        hidden_pos_states = self.activation_fn(self.fc1(hidden_pos_states))
        hidden_pos_states = self.activation_dropout(hidden_pos_states)

        if use_plugin == False:
            return hidden_pos_states

        # multi-head self-attention
        residual = hidden_pos_states
        hidden_pos_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_pos_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions
        )
        hidden_pos_states = self.dropout(hidden_pos_states)
        hidden_pos_states = residual + hidden_pos_states
        # hidden_pos_states = self.self_attention_layer_norm(hidden_pos_states)

        # check weight update
        # if self.prev_fc1_weight is not None:
        #         ic(torch.sum(torch.abs(self.fc1.weight - self.cur_fc1_weight)))
        # self.prev_fc1_weight = None if self.cur_fc1_weight is None else self.cur_fc1_weight.clone()
        # self.cur_fc1_weight = self.fc1.weight.clone()
        # if self.fc1.weight.grad is not None:
        #     ic(self.fc1.weight.requires_grad)
        #     ic(torch.sum(torch.abs(self.fc1.weight.grad)))

        # FFN with one linear layer
        residual = hidden_pos_states
        hidden_pos_states = self.activation_fn(self.fc2(hidden_pos_states))
        hidden_pos_states = self.activation_dropout(hidden_pos_states)
        hidden_pos_states = residual + hidden_pos_states
        hidden_pos_states = self.final_layer_norm(hidden_pos_states)

        # # output
        # if hidden_pos_states.dtype == torch.float16 and (
        #     torch.isinf(hidden_pos_states).any() or torch.isnan(hidden_pos_states).any()
        # ):
        #     clamp_value = torch.finfo(hidden_pos_states.dtype).max - 1000
        #     hidden_pos_states = torch.clamp(hidden_pos_states, min=-clamp_value, max=clamp_value)

        outputs = hidden_pos_states

        return outputs


class ExtendDecoderLayer(nn.Module):
    def __init__(self,
                 embed_dim,
                 num_heads,
                 attn_dropout,
                 fn_dropout,
                 act_dropout,
                 outer_dim
                 ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ffn_dim = embed_dim
        self.attn_dropout = attn_dropout
        self.outer_dim = outer_dim

        # self attention
        self.self_attn = BbBartAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout=self.attn_dropout,
            is_decoder=True
        )
        self.self_attention_layer_norm = nn.LayerNorm(self.ffn_dim)

        # cross attention
        self.encoder_attn = BbBartAttention(
            self.embed_dim,
            num_heads=self.num_heads,
            dropout=self.attn_dropout,
            is_decoder=True,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.ffn_dim)

        self.dropout = nn.Dropout(p=fn_dropout)
        self.activation_fn = ACT2FN["gelu"]
        self.activation_dropout = nn.Dropout(p=act_dropout)
        self.fc1 = nn.Linear(self.outer_dim, self.ffn_dim)
        self.fc2 = nn.Linear(self.ffn_dim, self.ffn_dim)
        self.final_layer_norm = nn.LayerNorm(self.ffn_dim)

    def forward(
            self,
            hidden_pos_states: torch.Tensor,
            attention_mask: torch.Tensor,
            encoder_attention_mask,
            encoder_hidden_pos_states: torch.Tensor,
            layer_head_mask: torch.Tensor,
            output_attentions: bool = False,
            past_key_value=None,
    ):
        # scale down, hidden_pos_states: (b, seq_len, out_dim) => (b, seq_len, embed_dim)
        hidden_pos_states = self.activation_fn(self.fc1(hidden_pos_states))
        hidden_pos_states = self.activation_dropout(hidden_pos_states)

        # multi-head self-attention
        residual = hidden_pos_states
        hidden_pos_states, attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_pos_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions
        )
        hidden_pos_states = self.dropout(hidden_pos_states)
        hidden_pos_states = residual + hidden_pos_states
        # hidden_pos_states = self.self_attention_layer_norm(hidden_pos_states)

        # Cross-Attention Block
        cross_attn_present_key_value = None
        cross_attn_weights = None
        if encoder_hidden_pos_states is not None:
            residual = hidden_pos_states

            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            hidden_pos_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
                hidden_states=hidden_pos_states,
                key_value_states=encoder_hidden_pos_states,
                attention_mask=encoder_attention_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
            )
            hidden_pos_states = self.dropout(hidden_pos_states)
            hidden_pos_states = residual + hidden_pos_states
            # hidden_pos_states = self.encoder_attn_layer_norm(hidden_pos_states)

            # # add cross-attn to positions 3,4 of present_key_value tuple
            # present_key_value = present_key_value + cross_attn_present_key_value

        # FFN with one linear layer
        residual = hidden_pos_states
        hidden_pos_states = self.activation_fn(self.fc2(hidden_pos_states))
        # hidden_pos_states = self.activation_dropout(hidden_pos_states)
        hidden_pos_states = residual + hidden_pos_states
        hidden_pos_states = self.final_layer_norm(hidden_pos_states)

        # output
        if hidden_pos_states.dtype == torch.float16 and (
                torch.isinf(hidden_pos_states).any() or torch.isnan(hidden_pos_states).any()
        ):
            clamp_value = torch.finfo(hidden_pos_states.dtype).max - 1000
            hidden_pos_states = torch.clamp(hidden_pos_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_pos_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class BbBartEncoderLayer(nn.Module):
    def __init__(self, config: BartConfig, tutaconfig):
        super().__init__()
        self.tutaconfig = tutaconfig
        self.embed_dim = config.d_model
        self.self_attn = BbBartAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

        self.plugin = ExtendEncoderLayer(
            embed_dim=self.embed_dim // config.encoder_attention_heads * self.tutaconfig.num_added_heads,
            num_heads=self.tutaconfig.num_added_heads,
            attn_dropout=config.attention_dropout,
            fn_dropout=config.dropout,
            act_dropout=config.activation_dropout,
            outer_dim=self.embed_dim + self.embed_dim // config.encoder_attention_heads * self.tutaconfig.num_added_heads
        )

        self.First = True

        self.prev_fc1_weight = None
        self.cur_fc1_weight = None

    def load_bart_plugin(self):
        self.plugin.self_attn.k_proj.weight.data = self.self_attn.k_proj.weight.data
        self.plugin.self_attn.v_proj.weight.data = self.self_attn.v_proj.weight.data
        self.plugin.self_attn.q_proj.weight.data = self.self_attn.q_proj.weight.data
        self.plugin.self_attn.out_proj.weight.data = self.self_attn.out_proj.weight.data

        self.plugin.self_attn.k_proj.bias = self.self_attn.k_proj.bias
        self.plugin.self_attn.v_proj.bias = self.self_attn.v_proj.bias
        self.plugin.self_attn.q_proj.bias = self.self_attn.q_proj.bias
        self.plugin.self_attn.out_proj.bias = self.self_attn.out_proj.bias

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor,
            layer_head_mask: torch.Tensor,
            output_attentions: bool = False,
            pos_row=None,
            pos_col=None,
            hidden_position_states: torch.Tensor = None,
            use_plugin=False
    ):
        """
        Args:
            hidden_states (:obj:`torch.FloatTensor`): input to the layer of shape `(seq_len, batch, embed_dim)`
            attention_mask (:obj:`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (:obj:`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
        """

        # bias config
        bias = pos_row is not None and pos_col is not None
        residual = hidden_states
        residual_ = hidden_states
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
            pos_row=pos_row,
            pos_col=pos_col,
            bias=bias
        )

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        if hidden_position_states is not None:
            hidden_states_concat = torch.cat((residual_, hidden_position_states), dim=-1)
            # hidden_states_concat = hidden_position_states
            plugin_out = self.plugin(
                hidden_pos_states=hidden_states_concat,
                attention_mask=attention_mask,
                layer_head_mask=layer_head_mask,
                output_attentions=output_attentions,
                use_plugin=use_plugin
            )
            hidden_position_states = plugin_out
            # check weight update
            # if self.prev_fc1_weight is not None:
            #     ic(torch.sum(torch.abs(self.plugin.fc1.weight - self.cur_fc1_weight)))
            # self.prev_fc1_weight = None if self.cur_fc1_weight is None else self.cur_fc1_weight.clone()
            # self.cur_fc1_weight = self.plugin.fc1.weight[:]
            # if self.plugin.fc1.weight.grad is not None:
            #     ic(self.plugin.fc1.weight.requires_grad)
            #     ic(torch.sum(torch.abs(self.plugin.fc1.weight.grad)))
            # if self.prev_fc1_weight is not None:
            #     ic(torch.sum(torch.abs(self.fc1.weight - self.cur_fc1_weight)))
            # self.prev_fc1_weight = None if self.cur_fc1_weight is None else self.cur_fc1_weight.clone()
            # self.cur_fc1_weight = self.fc1.weight.clone()

        if hidden_states.dtype == torch.float16 and (
                torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        if hidden_position_states is not None:
            outputs = (hidden_states, hidden_position_states)
        else:
            outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class BbBartDecoderLayer(nn.Module):
    def __init__(self, config: BartConfig, tutaconfig):
        super().__init__()
        self.tutaconfig = tutaconfig
        self.embed_dim = config.d_model

        self.self_attn = BbBartAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_attn = BbBartAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

        self.plugin = ExtendDecoderLayer(
            embed_dim=self.embed_dim // config.encoder_attention_heads * self.tutaconfig.num_added_heads,
            num_heads=self.tutaconfig.num_added_heads,
            attn_dropout=config.attention_dropout,
            fn_dropout=config.dropout,
            act_dropout=config.activation_dropout,
            outer_dim=self.embed_dim + self.embed_dim // config.encoder_attention_heads * self.tutaconfig.num_added_heads
            # + self.embed_dim +
        )
        self.prev_fc1_weight = None
        self.cur_fc1_weight = None

        # self.plugin = PlugLayer(config)

    def load_bart_plugin(self):
        self.plugin.self_attn.k_proj.weight.data = self.self_attn.k_proj.weight.data
        self.plugin.self_attn.v_proj.weight.data = self.self_attn.v_proj.weight.data
        self.plugin.self_attn.q_proj.weight.data = self.self_attn.q_proj.weight.data
        self.plugin.self_attn.out_proj.weight.data = self.self_attn.out_proj.weight.data

        self.plugin.self_attn.k_proj.bias = self.self_attn.k_proj.bias
        self.plugin.self_attn.v_proj.bias = self.self_attn.v_proj.bias
        self.plugin.self_attn.q_proj.bias = self.self_attn.q_proj.bias
        self.plugin.self_attn.out_proj.bias = self.self_attn.out_proj.bias

        self.First = True

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            layer_head_mask: Optional[torch.Tensor] = None,
            cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = True,
            pos_row=None,
            pos_col=None,
            encoder_hidden_position_states: torch.Tensor = None,
            hidden_position_states: torch.Tensor = None
    ):
        """
        Args:
            hidden_states (:obj:`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (:obj:`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            encoder_hidden_states (:obj:`torch.FloatTensor`): cross attention input to the layer of shape `(batch, seq_len, embed_dim)`
            encoder_attention_mask (:obj:`torch.FloatTensor`): encoder attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (:obj:`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            cross_attn_layer_head_mask (:obj:`torch.FloatTensor`): mask for cross-attention heads in a given layer of
                size `(decoder_attention_heads,)`.
            past_key_value (:obj:`Tuple(torch.FloatTensor)`): cached past key and value projection states
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
        """

        bias = pos_row is not None and pos_col is not None
        residual = hidden_states
        residual_ = hidden_states[:]

        # Self Attention
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # add present self-attn cache to positions 1,2 of present_key_value tuple
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
            pos_row=pos_row,
            pos_col=pos_col,
            bias=bias
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Cross-Attention Block
        cross_attn_present_key_value = None
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states

            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
            )
            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

            # add cross-attn to positions 3,4 of present_key_value tuple
            present_key_value = present_key_value + cross_attn_present_key_value

        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        if hidden_position_states is not None:
            hidden_states_concat = torch.cat((residual_.detach(), hidden_position_states), dim=-1)
            # hidden_states_concat = hidden_position_states
            plugin_out = self.plugin(
                hidden_pos_states=hidden_states_concat,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
                encoder_hidden_pos_states=encoder_hidden_position_states,
                layer_head_mask=layer_head_mask,
                output_attentions=output_attentions
            )
            hidden_position_states = plugin_out[0]
            # if self.prev_fc1_weight is not None:
            #     ic(torch.sum(torch.abs(self.plugin.fc2.weight - self.cur_fc1_weight)))
            # self.prev_fc1_weight = None if self.cur_fc1_weight is None else self.cur_fc1_weight.clone()
            # self.cur_fc1_weight = self.plugin.fc2.weight.clone()
        if hidden_position_states is not None:
            outputs = (hidden_states, hidden_position_states)
        else:
            outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class BbBartEncoder(BartEncoder):
    def __init__(self, config: BartConfig, tutaconfig, embed_tokens, row_weight=None, column_weight=None):
        super().__init__(config)

        self.tutaconfig = tutaconfig

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0
        self.add_res_layer = False
        self.add_res_layer_back = False

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)
        if row_weight is not None and column_weight is not None:
            self.row_weight = row_weight
            self.column_weight = column_weight
            # self.row_weight.weight.requires_grad = True
            # self.column_weight.weight.requires_grad = True

        self.addn_scale_down = nn.Linear(embed_dim,
                                         embed_dim // config.encoder_attention_heads * self.tutaconfig.num_added_heads)
        self.addn_scale_down_token = nn.Embedding(config.vocab_size,
                                                  embed_dim // config.encoder_attention_heads * self.tutaconfig.num_added_heads)

        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            embed_dim,
        )
        if tutaconfig.add_res_layer:
            self.add_res_layer = True
            self.res_att_layer = enc.Layer(tutaconfig)
            self.res_beta = tutaconfig.res_beta
        self.layers = nn.ModuleList([BbBartEncoderLayer(config, self.tutaconfig) for _ in range(config.encoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(embed_dim)

        self.init_weights()

        self.prev_fc1_weight = None
        self.cur_fc1_weight = None

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            pos_row=None,
            pos_col=None,
            check_pos_order=False,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        seq_position_states = self.embed_positions(input_shape)
        if check_pos_order:
            hidden_states = inputs_embeds + seq_position_states  # 768d
            hidden_position_states = self.addn_scale_down(hidden_states)  # 196d
            if self.tutaconfig.use_structure_pos and not self.tutaconfig.structure_pos_at_end:
                embed_row_pos = self.row_weight(pos_row)
                embed_col_pos = self.column_weight(pos_col)
                structure_position_states = embed_row_pos + embed_col_pos  # 196d, input row/col emb
                hidden_position_states += structure_position_states
            # check weight update
            # if self.prev_fc1_weight is not None:
            #     ic(torch.sum(torch.abs(self.addn_scale_down.weight - self.cur_fc1_weight)))
            # self.prev_fc1_weight = None if self.cur_fc1_weight is None else self.cur_fc1_weight.clone()
            # self.cur_fc1_weight = self.addn_scale_down.weight.clone()
        else:
            hidden_states = inputs_embeds + seq_position_states
            hidden_position_states = None
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            assert head_mask.size()[0] == (
                len(self.layers)
            ), f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
        for idx, encoder_layer in enumerate(self.layers):

            use_plugin = True if idx % 2 == 0 else False

            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                layer_outputs = (None, None)
            else:
                if getattr(self.config, "gradient_checkpointing", False) and self.training:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(encoder_layer),
                        hidden_states,
                        attention_mask,
                        (head_mask[idx] if head_mask is not None else None),
                        pos_row=pos_row,
                        pos_col=pos_col,
                        hidden_position_states=hidden_position_states,
                        use_plugin=use_plugin
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        output_attentions=output_attentions,
                        pos_row=pos_row,
                        pos_col=pos_col,
                        hidden_position_states=hidden_position_states,
                        use_plugin=use_plugin
                    )
                if hidden_position_states is not None:
                    hidden_states, hidden_position_states = layer_outputs[0], layer_outputs[1]
                    # hidden_position_states += hidden_states
                else:
                    hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)

        if hidden_position_states is not None:
            return BaseModelOutput(
                last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
            ), hidden_position_states
        else:
            return BaseModelOutput(
                last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
            ), hidden_position_states


class BbBartDecoder(BartDecoder):
    def __init__(self, config: BartConfig, tutaconfig, embed_tokens, row_weight=None, column_weight=None):
        super().__init__(config)
        self.tutaconfig = tutaconfig
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0
        embed_dim = config.d_model
        self.add_res_layer = False
        self.add_res_layer_back = False

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)

        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
        )
        if row_weight is not None and column_weight is not None:
            self.row_weight = row_weight
            self.column_weight = column_weight
            # self.row_weight = nn.Embedding(tutaconfig.row_size + 1, embed_dim, self.padding_idx)
            # self.row_weight_.weight.data.normal_(0, 1e-3)
            # self.column_weight = nn.Embedding(tutaconfig.column_size + 1, embed_dim, self.padding_idx)
        if tutaconfig.add_res_layer_back:
            self.add_res_layer_back = True
            from copy import deepcopy
            tutaconfig_ = deepcopy(tutaconfig)
            tutaconfig_.hidden_size = 768
            tutaconfig_.num_attention_heads = 12
            self.res_att_layer = enc.Layer(tutaconfig_)
            self.res_beta = tutaconfig.res_beta
        self.layers = nn.ModuleList([BbBartDecoderLayer(config, self.tutaconfig) for _ in range(config.decoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(config.d_model)

        self.addn_scale_down = nn.Linear(embed_dim,
                                         embed_dim // config.encoder_attention_heads * self.tutaconfig.num_added_heads)

        # self.addn_scale_down = nn.Linear(self.embed_dim, self.embed_dim // config.encoder_attention_heads * 2)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape, inputs_embeds.dtype, past_key_values_length=past_key_values_length
            ).to(self.device)

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            head_mask=None,
            cross_attn_head_mask=None,
            past_key_values=None,
            inputs_embeds=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            pos_row=None,
            pos_col=None,
            check_pos_order=False,
            position_states=None
    ):
        r"""
        Args:
            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using :class:`~transformers.BartTokenizer`. See
                :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__`
                for details.

                `What are input IDs? <../glossary.html#input-ids>`__
            attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                `What are attention masks? <../glossary.html#attention-mask>`__
            encoder_hidden_states (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, encoder_sequence_length, hidden_size)`, `optional`):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                of the decoder.
            encoder_attention_mask (:obj:`torch.LongTensor` of shape :obj:`(batch_size, encoder_sequence_length)`, `optional`):
                Mask to avoid performing cross-attention on padding tokens indices of encoder input_ids. Mask values
                selected in ``[0, 1]``:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                `What are attention masks? <../glossary.html#attention-mask>`__
            head_mask (:obj:`torch.Tensor` of shape :obj:`(decoder_layers, decoder_attention_heads)`, `optional`):
                Mask to nullify selected heads of the attention modules. Mask values selected in ``[0, 1]``:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            cross_attn_head_mask (:obj:`torch.Tensor` of shape :obj:`(decoder_layers, decoder_attention_heads)`, `optional`):
                Mask to nullify selected heads of the cross-attention modules in the decoder to avoid performing
                cross-attention on hidden heads. Mask values selected in ``[0, 1]``:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (:obj:`tuple(tuple(torch.FloatTensor))`, `optional`, returned when ``use_cache=True`` is passed or when ``config.use_cache=True``):
                Tuple of :obj:`tuple(torch.FloatTensor)` of length :obj:`config.n_layers`, with each tuple having 2
                tensors of shape :obj:`(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional
                tensors of shape :obj:`(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see :obj:`past_key_values` input) to speed up sequential
                decoding.

                If :obj:`past_key_values` are used, the user can optionally input only the last
                :obj:`decoder_input_ids` (those that don't have their past key value states given to this model) of
                shape :obj:`(batch_size, 1)` instead of all :obj:`decoder_input_ids`` of shape :obj:`(batch_size,
                sequence_length)`.
            inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
                Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded
                representation. This is useful if you want more control over how to convert :obj:`input_ids` indices
                into associated vectors than the model's internal embedding lookup matrix.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
            output_hidden_states (:obj:`bool`, `optional`):
                Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors
                for more detail.
            return_dict (:obj:`bool`, `optional`):
                Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )

        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _expand_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])

        # embed positions
        seq_position_states = self.embed_positions(input_shape, past_key_values_length)

        if check_pos_order:
            hidden_states = inputs_embeds + seq_position_states  # 768d
            hidden_position_states = self.addn_scale_down(hidden_states)  # 196d
            if self.tutaconfig.use_structure_pos and not self.tutaconfig.structure_pos_at_end:
                embed_row_pos = self.row_weight(pos_row)
                embed_col_pos = self.column_weight(pos_col)
                structure_position_states = embed_row_pos + embed_col_pos  # 196d, input row/col emb
                hidden_position_states += structure_position_states
            encoder_hidden_position_states = position_states[:]
        else:
            hidden_states = inputs_embeds + seq_position_states
            encoder_hidden_position_states = None
            hidden_position_states = None
        hidden_states = self.layernorm_embedding(hidden_states)

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]):
            if attn_mask is not None:
                assert attn_mask.size()[0] == (
                    len(self.layers)
                ), f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if getattr(self.config, "gradient_checkpointing", False) and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
                        "`use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, use_cache)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None,
                    past_key_value=None,
                    output_attentions=None,
                    use_cache=None,
                    pos_row=pos_row,
                    pos_col=pos_col,
                    encoder_hidden_position_states=encoder_hidden_position_states,
                    hidden_position_states=hidden_position_states
                )
            else:
                # ic(encoder_attention_mask) 全部是None
                # ic(head_mask)
                # ic(cross_attn_head_mask)
                # ic(past_key_value)

                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    cross_attn_layer_head_mask=(
                        cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None
                    ),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    pos_row=pos_row,
                    pos_col=pos_col,
                    encoder_hidden_position_states=encoder_hidden_position_states,
                    hidden_position_states=hidden_position_states
                )
            if hidden_position_states is not None:
                hidden_states, hidden_position_states = layer_outputs[0], layer_outputs[1]
                # hidden_position_states += hidden_states
            else:
                hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # # add hidden_position_states
        # device = hidden_states.device
        # table_mask = torch.ones_like(hidden_states).to(device)
        # table_mask[pos_row == 256, :] = 0
        # # ic(table_mask[0, :10, 10])
        # # ic(torch.mean(pos_row * 1.0), torch.sum(table_mask))
        # table_mask = table_mask.contiguous().detach()
        # hidden_states += hidden_position_states * table_mask

        # additional layer in the back: # TODO
        if self.add_res_layer_back:
            res_attention_mask = (1 - attention_mask) * -10000.0 + attention_mask
            # ic(hidden_states.size())
            F_hidden_states = self.res_att_layer(hidden_states, res_attention_mask)

            # create mask for non-structure token(text, paragraph)
            device = hidden_states.device
            table_mask = torch.ones_like(hidden_states).to(device)
            # table_mask[pos_row == 256, :] = 0
            # ic(table_mask[0, :10, 10])
            # ic(torch.mean(pos_row * 1.0), torch.sum(table_mask))
            table_mask = table_mask.contiguous().detach()
            F_hidden_states = F_hidden_states * table_mask

            hidden_states = hidden_states + F_hidden_states * self.res_beta

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        ), hidden_position_states


class BbForBart(BartModel):
    def __init__(self, config: BartConfig, tutaconfig):
        super().__init__(config)

        self.tutaconfig = tutaconfig
        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)
        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        self.row_weight = nn.Embedding(
            tutaconfig.row_size + 1,
            embed_dim // config.encoder_attention_heads * self.tutaconfig.num_added_heads,
            self.padding_idx
        )
        self.column_weight = nn.Embedding(
            tutaconfig.column_size + 1,
            embed_dim // config.encoder_attention_heads * self.tutaconfig.num_added_heads,
            self.padding_idx
        )

        self.encoder = BbBartEncoder(config, tutaconfig, self.shared, self.row_weight, self.column_weight)
        self.decoder = BbBartDecoder(config, tutaconfig, self.shared, self.row_weight, self.column_weight)
        self.args = tutaconfig

        self.init_weights()

    def get_input_embeddings(self, value):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            pos_row=None,
            pos_col=None,
    ):
        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        # shift pos_row and pos_col
        decoder_pos_row = shift_tokens_right(
            pos_row, 256, 256
        )
        decoder_pos_col = shift_tokens_right(
            pos_col, 256, 256
        )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        enc_input_embeds = self.shared(input_ids) * self.embed_scale
        dec_input_embeds = self.shared(decoder_input_ids) * self.embed_scale

        if encoder_outputs is None:
            encoder_outputs, encoder_pos_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                pos_row=pos_row,
                pos_col=pos_col,
                check_pos_order=self.args.check_pos_order
            )
            # ic(torch.sum(encoder_pos_outputs))

        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs, decoder_pos_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            pos_row=decoder_pos_row,
            pos_col=decoder_pos_col,
            check_pos_order=self.args.check_pos_order,
            position_states=encoder_pos_outputs
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        last_hidden_states = decoder_outputs.last_hidden_state
        if self.tutaconfig.check_pos_order and self.tutaconfig.structure_pos_at_end:
            embed_row_pos = self.row_weight(pos_row)
            embed_col_pos = self.column_weight(pos_col)
            structure_position_states = embed_row_pos + embed_col_pos  # 196d, input row/col emb
            decoder_pos_outputs += structure_position_states
        if decoder_pos_outputs is not None:
            last_hidden_states = torch.cat((last_hidden_states, decoder_pos_outputs), dim=-1)

        return Seq2SeqModelOutput(
            last_hidden_state=last_hidden_states,
            past_key_values=None,  # decoder_outputs.past_key_values,
            decoder_hidden_states=None,  # decoder_outputs.hidden_states,
            decoder_attentions=None,  # decoder_outputs.attentions,
            cross_attentions=None,  # decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


BACKBONES = {
    "tuta": BbForTuta,
    "base": BbForBase,
    "tuta_explicit": BbForTutaExplicit,
    "tuta_formula": BbForTuta,
    "tuta_fp": BbForTuta,
    "tuta_formula_w_clc": BbForTuta,
    "tuta_formula_mlm": BbForTuta,
    "tuta_formula_combine": BbForTuta,
    'tuta_formula_cs': BbForTuta,
    'tuta_context_aug': BbForTuta,
    'tuta_formula_v5': BbForTuta,
    'tuta_entity_link': BbForTuta,
    'tuta_wtq_qa': BbForTuta,
    'bart_entity_link': BbForBart,
}
