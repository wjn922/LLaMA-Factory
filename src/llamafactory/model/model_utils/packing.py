# Copyright 2025 Musab Gultekin and the LlamaFactory team.
#
# This code is based on the Musab Gultekin's functionary library.
# https://github.com/MeetKai/functionary/blob/main/functionary/train/packing/monkey_patch_packing.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# MIT License
#
# Copyright (c) 2023 Musab Gultekin
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from ...extras import logging


if TYPE_CHECKING:
    from ...hparams import ModelArguments
    from transformers import PretrainedConfig


from typing import Optional
from transformers.utils.deprecation import deprecate_kwarg
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.cache_utils import Cache
from transformers.processing_utils import Unpack
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLVisionModel,
    Qwen3VLModel,
    apply_rotary_pos_emb,
)

from flash_attn.flash_attn_interface import flash_attn_varlen_func




logger = logging.get_logger(__name__)


def get_seqlens_in_batch(attention_mask: "torch.Tensor") -> "torch.Tensor":
    r"""Get the sequence lengths in the current batch.

    e.g.
    ```python
    # input
    [
        [1, 1, 2, 2, 2, 0],
        [1, 2, 2, 3, 3, 3],
    ]
    # output
    [2, 3, 1, 2, 3]
    ```
    """
    bsz = attention_mask.size(0)
    dtype, device = attention_mask.dtype, attention_mask.device
    max_num = torch.max(attention_mask).item()
    counts: torch.Tensor = torch.zeros((bsz, max_num), dtype=dtype, device=device)
    for i in range(max_num):
        counts[:, i] = torch.sum(attention_mask == (i + 1), dim=-1)

    counts = counts.flatten()
    seqlens = counts[counts.nonzero().squeeze(dim=-1)]
    return seqlens


def get_unpad_data(attention_mask: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor", int]:
    r"""Prepare the indices and seqlens for flash attn varlen function.

    Returns:
        indices: indices of non-masked tokens from the flattened sequence.
        cu_seqlens: the cumulative sequence lengths in the current batch, always starts from 0.
        max_seqlen_in_batch: the largest seqlen in the current batch.

    e.g.
    ```python
    # input
    [
        [1, 1, 2, 2, 2, 0],
        [1, 2, 2, 3, 3, 3],
    ]
    # output
    [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11]
    [0, 2, 5, 6, 8, 11]
    3
    ```

    """
    # when enable packing, the batch size can be larger than 1
    # here, we flatten all the batches into one single sequence
    seqlens_in_batch = get_seqlens_in_batch(attention_mask)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return indices, cu_seqlens, max_seqlen_in_batch


# qwen3-vl 
def flash_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    sliding_window: Optional[int] = None,
    softcap: Optional[float] = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    if kwargs.get("output_attentions", False) or kwargs.get("head_mask") is not None:
        logger.warning_once(
            "`flash_attention_2` does not support `output_attentions=True` or `head_mask`."
            " Please set your attention to `eager` if you want any of these features."
        )
    
    # This is before the transpose
    seq_len = query.shape[2]

    if any(dim == 0 for dim in query.shape):
        raise ValueError(
            "Tensor query has shape with a zero dimension.\n"
            "FlashAttention does not support inputs with dim=0.\n"
            "Please check your input shapes or use SDPA instead."
        )
    # FA2 uses non-transposed inputs
    # batch, head, seq_len, dim
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    # batch, seqlen, head, dim

    # In PEFT, usually we cast the layer norms in float32 for training stability reasons
    # therefore the input hidden states gets silently casted in float32. Hence, we need
    # cast them back in the correct dtype just to be sure everything works as expected.
    # This might slowdown training & inference so it is recommended to not cast the LayerNorms
    # in fp32. (usually our RMSNorm modules handle it correctly)
    target_dtype = None
    if query.dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        elif hasattr(module.config, "_pre_quantization_dtype"):
            target_dtype = module.config._pre_quantization_dtype
        else:
            target_dtype = next(layer for layer in module.modules() if isinstance(layer, torch.nn.Linear)).weight.dtype
        
        if target_dtype is not None:
            query = query.to(target_dtype)
            key = key.to(target_dtype)
            value = value.to(target_dtype)

    # Convert LLaMA-Factory's attention_mask format to cu_seqlens
    # attention_mask format: [1, 1, 1, 2, 2, 3, 0, 0, ...] with shape [bsz, seq_len]
    # Need to convert to cu_seqlens and get unpad indices
    batch_size = query.shape[0]
    
    if batch_size == 1:
        # Single batch case: use get_unpad_data to convert attention_mask to cu_seqlens
        indices, cu_seqlens, max_seqlen = get_unpad_data(attention_mask)
        
        # Flatten and index query, key, value to remove padding
        # Shape: [batch, seqlen, head, dim] -> [total_tokens, head, dim]
        query = query.squeeze(0)[indices]  # Remove batch dim and index
        key = key.squeeze(0)[indices]
        value = value.squeeze(0)[indices]
    else:
        # Multi-batch case (shouldn't happen with packing, but handle it)
        raise ValueError("Packing should result in batch_size=1 after packing")

    attn_output = flash_attn_varlen_func(
        query,
        key,
        value,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        dropout_p=dropout,
        causal=True,
    )

    # Restore the original shape by padding back
    # Create output tensor with original sequence length
    output_shape = (batch_size, attention_mask.shape[1], attn_output.shape[-2], attn_output.shape[-1])
    full_attn_output = torch.zeros(
        output_shape, dtype=attn_output.dtype, device=attn_output.device
    )
    
    # Place the computed attention output back to non-masked positions
    full_attn_output.squeeze(0)[indices] = attn_output
    
    return full_attn_output, None



@deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
def qwen3vl_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_values: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

    attn_output, attn_weights = flash_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def return_mask(
    config,
    input_embeds,
    attention_mask,
    cache_position,
    past_key_values,
    position_ids,
    **kwargs
):
    return attention_mask


def configure_packing(config: "PretrainedConfig", model_args: "ModelArguments", is_trainable: bool) -> None:
    if not is_trainable or not model_args.block_diag_attn:
        return

    # neat_packing = True -> block_diag_attn = True
    model_type = getattr(config, "model_type", None)
    if model_type == "qwen3_vl":
        assert model_args.flash_attn == "fa2", "Qwen3-VL requires flash_attn='fa2' when using block diagonal attention."
        # flash attn monkey patch for packing
        transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLTextAttention.forward = (
            qwen3vl_forward
        )
        transformers.models.qwen3_vl.modeling_qwen3_vl.create_causal_mask = (
            return_mask
        )
        logger.info_rank0("Using block diagonal attention for sequence packing for Qwen3-VL.")
    else:
        from ...extras.packages import is_transformers_version_greater_than
        if is_transformers_version_greater_than("4.53.0"):
            raise ValueError("Neat packing is incompatible with transformers>=4.53.0.")

        import transformers.modeling_flash_attention_utils

        transformers.modeling_flash_attention_utils._get_unpad_data = get_unpad_data
        logger.info_rank0("Using block diagonal attention for sequence packing without cross-attention.")


