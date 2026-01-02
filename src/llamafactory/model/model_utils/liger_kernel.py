# Copyright 2025 the LlamaFactory team.
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

import inspect
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from ...extras import logging


if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from ...hparams import ModelArguments


logger = logging.get_logger(__name__)


def _patch_qwen3vl_vision_patch_embed() -> None:
    """
    Patch the Qwen3VLVisionPatchEmbed forward method to avoid slow conv3d operations.
    This replaces conv3d with linear operations for better performance.
    """
    try:
        from transformers.models.qwen3_vl import modeling_qwen3_vl
    except ImportError:
        logger.warning_rank0("Cannot import Qwen3VL modeling, skipping PatchEmbed patch.")
        return

    def patch_embed_forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Patch the forward of the Qwen3VLVisionPatchEmbed.
        """
        target_dtype = self.proj.weight.dtype
        proj_weight = self.proj.weight
        proj_bias = self.proj.bias
        # compute in fp32 for numerical stability (even under outer autocast), then cast back
        with torch.amp.autocast(device_type="cuda", enabled=False):
            hidden_states_fp32 = hidden_states.float()
            weight_fp32 = proj_weight.view(self.embed_dim, -1).float()
            bias_fp32 = proj_bias.float() if proj_bias is not None else None
            hidden_states = F.linear(hidden_states_fp32, weight_fp32, bias_fp32)
        hidden_states = hidden_states.to(dtype=target_dtype)
        return hidden_states

    modeling_qwen3_vl.Qwen3VLVisionPatchEmbed.forward = patch_embed_forward
    logger.info_rank0("Applied PatchEmbed optimization for Qwen3VL.")


def apply_liger_kernel(
    config: "PretrainedConfig",
    model_args: "ModelArguments",
    is_trainable: bool,
    require_logits: bool,
) -> None:
    if not is_trainable or not model_args.enable_liger_kernel:
        return

    model_type = getattr(config, "model_type", None)
    if model_type == "gemma":
        from liger_kernel.transformers import apply_liger_kernel_to_gemma as apply_liger_kernel
    elif model_type == "gemma2":
        from liger_kernel.transformers import apply_liger_kernel_to_gemma2 as apply_liger_kernel
    elif model_type == "gemma3":
        from liger_kernel.transformers import apply_liger_kernel_to_gemma3 as apply_liger_kernel
    elif model_type == "gemma3_text":
        from liger_kernel.transformers import apply_liger_kernel_to_gemma3_text as apply_liger_kernel
    elif model_type == "glm4":
        from liger_kernel.transformers import apply_liger_kernel_to_glm4 as apply_liger_kernel
    elif model_type == "glm4v":
        from liger_kernel.transformers import apply_liger_kernel_to_glm4v as apply_liger_kernel
    elif model_type == "granite":
        from liger_kernel.transformers import apply_liger_kernel_to_granite as apply_liger_kernel
    elif model_type == "llama":
        from liger_kernel.transformers import apply_liger_kernel_to_llama as apply_liger_kernel
    elif model_type == "llava":
        from liger_kernel.transformers import apply_liger_kernel_to_llava as apply_liger_kernel
    elif model_type == "mistral":
        from liger_kernel.transformers import apply_liger_kernel_to_mistral as apply_liger_kernel
    elif model_type == "mixtral":
        from liger_kernel.transformers import apply_liger_kernel_to_mixtral as apply_liger_kernel
    elif model_type == "mllama":
        from liger_kernel.transformers import apply_liger_kernel_to_mllama as apply_liger_kernel
    elif model_type == "olmo2":
        from liger_kernel.transformers import apply_liger_kernel_to_olmo2 as apply_liger_kernel
    elif model_type == "paligemma":
        from liger_kernel.transformers import apply_liger_kernel_to_paligemma as apply_liger_kernel
    elif model_type == "phi3":
        from liger_kernel.transformers import apply_liger_kernel_to_phi3 as apply_liger_kernel
    elif model_type == "qwen2":
        from liger_kernel.transformers import apply_liger_kernel_to_qwen2 as apply_liger_kernel
    elif model_type == "qwen2_vl":
        from liger_kernel.transformers import apply_liger_kernel_to_qwen2_vl as apply_liger_kernel
    elif model_type == "qwen2_5_vl":
        from liger_kernel.transformers import apply_liger_kernel_to_qwen2_5_vl as apply_liger_kernel
    elif model_type == "qwen3":
        from liger_kernel.transformers import apply_liger_kernel_to_qwen3 as apply_liger_kernel
    elif model_type == "qwen3_vl":
        from liger_kernel.transformers import apply_liger_kernel_to_qwen3_vl as apply_liger_kernel
    elif model_type == "qwen3_moe":
        from liger_kernel.transformers import apply_liger_kernel_to_qwen3_moe as apply_liger_kernel
    elif model_type == "gpt_oss":
        try:
            from liger_kernel.transformers import apply_liger_kernel_to_gpt_oss as apply_liger_kernel
        except ImportError:
            logger.warning_rank0("Please install liger-kernel from https://github.com/Comet0322/Liger-Kernel.")
            return
    else:
        logger.warning_rank0("Current model does not support liger kernel.")
        return

    if require_logits and "fused_linear_cross_entropy" in inspect.signature(apply_liger_kernel).parameters:
        logger.info_rank0("Current training stage does not support chunked cross entropy.")
        kwargs = {"fused_linear_cross_entropy": False, "cross_entropy": True}
    else:
        kwargs = {}

    apply_liger_kernel(**kwargs)
    logger.info_rank0("Liger kernel has been applied to the model.")

    # Apply Qwen3VL specific patches
    if model_type == "qwen3_vl":
        _patch_qwen3vl_vision_patch_embed()
