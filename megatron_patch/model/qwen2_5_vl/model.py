# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import logging
from collections import namedtuple
from typing import List

import torch

from megatron.core import InferenceParams, parallel_state, tensor_parallel
from megatron.core.transformer import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from .transformer_config import Qwen2VLTransformerConfig
from megatron.core.packed_seq_params import PackedSeqParams

from .visionmodel import Qwen2_5VisionModel
from .utils import get_batch_on_this_cp_rank
from megatron_patch.model.qwen2_vl.gpt_model import GPTModel


def print_current_rank(point:str):
    dp_rank = parallel_state.get_data_parallel_rank()
    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    cp_rank = parallel_state.get_context_parallel_rank()
    print(f"{point}-dp_rank: {dp_rank}, tp_rank: {tp_rank}, cp_rank: {cp_rank}")

class AllGatherVisionEmbeddings(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, seqlens_on_cp_ranks):
        outputs = []
        for i in range(len(seqlens_on_cp_ranks)):
            o =torch.zeros((seqlens_on_cp_ranks[i].sum(), *input.shape[1:]), 
                                       device=input.device, 
                                       dtype=input.dtype, 
                                       layout=input.layout)
            outputs.append(o)
        torch.distributed.all_gather(outputs, input, group=parallel_state.get_context_parallel_group())
        cp_rank = parallel_state.get_context_parallel_rank()
        ctx.cp_rank = cp_rank
        # ctx.seqlens_on_cp_ranks = seqlens_on_cp_ranks
        ctx.save_for_backward(*seqlens_on_cp_ranks)

        output = torch.cat(outputs, dim=0)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        cp_rank = ctx.cp_rank
        seqlens_on_cp_ranks = ctx.saved_tensors
        # seqlens_on_cp_ranks = ctx.seqlens_on_cp_ranks
        start_idx = torch.cat(seqlens_on_cp_ranks[:cp_rank]).sum() if cp_rank != 0 else 0
        end_idx = start_idx + seqlens_on_cp_ranks[cp_rank].sum()
        grad_output = grad_output[start_idx:end_idx]
        return grad_output, None


# Note: This is under development and may be missing features.
class Qwen2_5VLModel(MegatronModule):
    """Qwen2.5VL multi-modal model.

    Args:
        language_transformer_config (TransformerConfig): Transformer config for the language model.
        language_transformer_layer_spec (ModuleSpec): Specifies module to use for transformer layers of the language model.
        language_vocab_size (int): Language model vocabulary size.
        language_max_sequence_length (int): Language model maximum sequence length. This is used for positional embedding.
        vision_transformer_config (TransformerConfig): Transformer config for the vision model.
        vision_transformer_layer_spec (ModuleSpec): Specifies module to use for transformer layers of the vision model.
        drop_vision_class_token (bool): Drop vision class token(s) before input to the language model.
        vision_projection_config (TransformerConfig): Config for the projection from vision model outputs to language model inputs.
        vision_projection_layer_spec (ModuleSpec): Specifies the module to use for the vision projection.
        vision_projection_type (str): Type of the vision projection to use. Default is a 2-layer MLP.
        allow_missing_vision_projection_checkpoint (bool): Allow vision projection weights to be missing when loading a checkpoint. Default False.
        parallel_output (bool): Do not gather the outputs, keep them split across tensor parallel ranks. This is typically True for training and False for inference.
        language_position_embedding_type (str): Position embedding type to use in the language model. Default learned absolute.
        language_rotary_percent (float): Percent of rotary dimension to use for rotary position embeddings in the language model. Defaults to 1.0.
        pre_process (bool): Include the embedding layer in the gpt decoder (used with pipeline parallelism). Defaults to True.
        post_process (bool): Include an output layer and a layernorm in the gpt decoder (used with pipeline parallelism). Defaults to True.
        add_encoder (bool): Construct the encoder module (used with pipeline parallelism). Defaults to True. When we use pipelining, the encoder
            will live on only a subset of the pipeline stages (specifically, only the first stage).
        add_decoder (bool): Construct the decoder module (used with pipeline parallelism). Defaults to True. When we use pipelining, the decoder
            will live on only a subset of the pipeline stages (specifically, every stage after the first one).
        img_h (int): The height of each image that the ViT will see.
        img_w (int): The width of each image that the ViT will see.
        patch_dim (int): The size of each patch side.
        img_embedding_idx (int): Index in the language_embeddings tensor where image_embeddings should be inserted. Defaults to 0.
    """

    def __init__(
        self,
        language_transformer_config: Qwen2VLTransformerConfig,
        language_transformer_layer_spec: ModuleSpec,
        language_vocab_size: int,
        language_max_sequence_length: int,
        vision_transformer_config: TransformerConfig,
        vision_transformer_layer_spec: ModuleSpec,
        drop_vision_class_token: bool,
        vision_projection_config: TransformerConfig,
        vision_projection_layer_spec: ModuleSpec,
        vision_projection_type: str = "mlp",

        allow_missing_vision_projection_checkpoint: bool = False,
        parallel_output: bool = True,
        language_position_embedding_type: str = 'rope',
        language_rotary_percent: float = 1.0,
        pre_process: bool = True,
        post_process: bool = True,
        add_encoder: bool = True,
        add_decoder: bool = True,
        language_rotary_base: int = 10000,
        fp16_lm_cross_entropy: bool = False,
        language_share_embeddings_and_output_weights: bool=False
    ) -> None:
        super().__init__(config=language_transformer_config)

        logging.getLogger(__name__).warning(
            "Qwen2VL model is under development and may be missing features."
        )

        self.pre_process = pre_process
        self.post_process = post_process
        self.add_encoder = add_encoder
        self.add_decoder = add_decoder
        
        self.encoder_hidden_state = None
        self.vision_model = None
        self.vision_projection = None
        self.language_model = None

        self.square_merge_size = vision_projection_config.ffn_hidden_size // vision_transformer_config.hidden_size
        self.context_parallel_lm = language_transformer_config.context_parallel_size
        self.sequence_parallel_lm = language_transformer_config.sequence_parallel
        self.tensor_model_parallel_size_lm = language_transformer_config.tensor_model_parallel_size

        # This attribute is needed to check if an all-reduce is required
        # on the word embeddings inside `finalize_model_grads._allreduce_word_embedding_grads`.
        self.share_embeddings_and_output_weights = False
        if self.pre_process:
            self.vision_model = Qwen2_5VisionModel(
                vision_transformer_config,
                vision_transformer_layer_spec,
                vision_projection_config,
                vision_projection_layer_spec,
                projection_type=vision_projection_type,
                pre_process=True,
                post_process=True
            )

        self.language_model = GPTModel(
            config=language_transformer_config,
            transformer_layer_spec=language_transformer_layer_spec,
            vocab_size=language_vocab_size,
            max_sequence_length=language_max_sequence_length,
            parallel_output=parallel_output,
            position_embedding_type=language_position_embedding_type,
            rotary_percent=language_rotary_percent,
            pre_process=self.pre_process,
            post_process=self.post_process,
            rotary_base=language_rotary_base,
            mrope_section=language_transformer_config.mrope_section,
            fp16_lm_cross_entropy=fp16_lm_cross_entropy,
            share_embeddings_and_output_weights=language_share_embeddings_and_output_weights
        )
        self.share_embeddings_and_output_weights = (
            self.language_model.share_embeddings_and_output_weights
        )

    def shared_embedding_or_output_weight(self):
        """This is a convenience method to surface the language model's word embeddings, which is
        necessary for `finalize_model_grads._allreduce_word_embedding_grads`."""
        if self.add_decoder:
            return self.language_model.shared_embedding_or_output_weight()
        return None

    def set_input_tensor(self, input_tensor) -> None:
        # This is usually handled in schedules.py but some inference code still
        # gives us non-lists or None
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        assert len(input_tensor) == 1, 'input_tensor should only be length 1 for Qwen2VL'
        
        if self.pre_process:
            self.encoder_hidden_state = input_tensor[0]
        else:
            self.language_model.set_input_tensor(input_tensor[0])

    def freeze(
        self, freeze_language_model: bool, freeze_vision_model: bool, freeze_vision_projection: bool
    ):
        """Freeze model modules.

        Make specific modules non-trainable by setting requires_grad to False for the module's parameters.

        Args:
            freeze_language_model (bool): Freeze the language model module.
            freeze_vision_model (bool): Freeze the vision model module.
            freeze_vision_projection (bool): Freeze the vision projection module.
        """
        modules = []
        if freeze_language_model and self.language_model is not None:
            modules.append(self.language_model)
        if freeze_vision_model and self.vision_model is not None:
            modules.append(self.vision_model)
        if freeze_vision_projection and self.vision_projection is not None:
            modules.append(self.vision_projection)

        for module in modules:
            for param in module.parameters():
                param.requires_grad = False

    def _get_vision_cp_data(self, vision_data, vision_grid_thw):
        """Get vision data and grid_thw for context parallelism.
        
        Returns:
            vision_data (torch.Tensor): Vision data of shape [total_thw_size, n_features].
            vision_grid_thw (torch.Tensor): Vision grid_thw of shape [total_thw_size, 3].
            seqlens_list (list of torch.Tensor): List of seqlens of the vision data in each context parallel rank, for the all gather after vision encoder.
        """
        # we use the context parallelism size and context parallel group of LLM for vision model.
        # we only divide the number of images in each context parallel rank.
        cp_size = self.language_model.config.context_parallel_size
        cp_rank = parallel_state.get_context_parallel_rank()
        if cp_size == 1 or not getattr(self.vision_model.config, 'enable_context_parallelism', False):
            return vision_data, vision_grid_thw, None

        img_num = vision_grid_thw.shape[0]
        img_num_per_rank = (img_num + cp_size - 1) // cp_size
        seqlens = torch.repeat_interleave(vision_grid_thw[:, 1] * vision_grid_thw[:, 2], vision_grid_thw[:, 0])
        vision_grid_thw_list = []
        vision_data_list = []
        seqlens_list = []
        for i in range(cp_size):
            start_idx = i * img_num_per_rank
            end_idx = min(start_idx + img_num_per_rank, img_num)
            vision_grid_thw_list.append(vision_grid_thw[start_idx:end_idx])
            seqlens_list.append(seqlens[start_idx:end_idx])
            data_start_idx = seqlens[:start_idx].sum()
            data_end_idx = seqlens[:end_idx].sum()
            vision_data_list.append(vision_data[data_start_idx:data_end_idx])
        new_vision_grid_thw = vision_grid_thw_list[cp_rank]
        new_vision_data = vision_data_list[cp_rank]
        new_seqlens_list = [t // self.square_merge_size for t in seqlens_list]
        return new_vision_data, new_vision_grid_thw, new_seqlens_list

    def _process_embedding_token_parallel(
        self, combined_embeddings, labels, loss_mask, packed_seq_params
    ):
        """Processes the input data for model parallelism support.

        When using sequence parallelism (SP) or context parallelism (CP), the sequence is sharded
        across different GPUs. This function performs the sharding and distributes the sequence
        across GPUs for SP and CP

        Context Parallelism is a feature that helps improve memory efficiency for
        long sequence training by distributing sequence across CP ranks.
        It requires token length to be divisible by (CP size *2) to ensure proper load balance.

        Sequence Parallelism is a feature that helps improve memory efficiency for
        long sequence training by distributing sequence across TP ranks.
        It requires token length to be divisible by TP size.

        Returns:
            combined_embeddings (torch.Tensor): image and text embeddings combined and distributed. [S, B, H]
            new_labels (torch.Tensor): Distributed labels for image and text positions. [S, B]
            packed_seq_params (PackedSeqParams): Dict with padded token information.

        """

        # No pre or post processing needed with PP middle chunks.
        if not self.pre_process and not self.post_process:
            return combined_embeddings, labels, loss_mask, packed_seq_params

        shard_factor = seq_dim = None
        if self.pre_process:
            if self.context_parallel_lm > 1 and self.sequence_parallel_lm:
                shard_factor = self.tensor_model_parallel_size_lm * self.context_parallel_lm * 2
                # seq_dim = 1
            elif self.context_parallel_lm > 1:
                shard_factor = self.context_parallel_lm * 2
                # seq_dim = 1
            elif self.sequence_parallel_lm:
                shard_factor = self.tensor_model_parallel_size_lm
                # seq_dim = 0

            seq_dim = 0
            assert (
                combined_embeddings.shape[seq_dim] % shard_factor == 0
            ), f"Sequence length should be divisible by {shard_factor} for \
                Sequence/Context parallelism"
            if self.sequence_parallel_lm and self.tp_comm_overlap_lm:
                assert (
                    combined_embeddings.shape[seq_dim] == self._language_max_sequence_length
                ), f"TP Comm overlap either requires Vision+Text token length \
                == language_max_sequence_length"

        if self.context_parallel_lm > 1:
            batch = dict()
            if self.pre_process:
                batch["combined_embeddings"] = combined_embeddings
            if self.post_process:
                batch["labels"] = labels
                batch["loss_mask"] = loss_mask
            # Distribute sequence across CP ranks
            if packed_seq_params is None or packed_seq_params.qkv_format == 'sbhd':
                batch = get_batch_on_this_cp_rank(batch)
            else:
                raise NotImplementedError("THD format data is not supported yet")

            if self.pre_process:
                combined_embeddings = batch["combined_embeddings"]  # [S/CP, B, H]

            if self.post_process:
                new_labels = batch["labels"]
                new_loss_mask = batch["loss_mask"]

        if self.sequence_parallel_lm and self.pre_process:
            combined_embeddings = tensor_parallel.scatter_to_sequence_parallel_region(
                combined_embeddings
            )  # [S/(CP*TP),B,H]

        return combined_embeddings, new_labels, new_loss_mask, packed_seq_params

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        vision_data: torch.Tensor = None,
        vision_grid_thw: torch.Tensor = None,
        video_start_index: int = -1,
        image_input_mask: torch.Tensor = None,
        video_input_mask: torch.Tensor = None,
        loss_mask: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
    ) -> torch.Tensor:
        """Forward function of the Qwen2VL model.

        Args:
            image_data (torch.Tensor): input image of shape [total_thw_size, n_features].
            input_ids (torch.Tensor): input text ids [batch, text_seq_len].
            position_ids (torch.Tensor): input text position ids [batch, text_seq_len].
            attention_mask (torch.Tensor): attention mask for the language model [batch, 1, combined_seq_len, combined_seq_len].
            labels (torch.Tensor): Optional target text labels [batch, combined_seq_len].
            inference_params (InferenceParams): Inference-time parameters including KV cache.

            video_start_index:
                0 -- all video
                len(video_seq) -- all image
                others -- mixture
            *_input_mask: should not be None in the first PP stage
        Returns:
            output (torch.Tensor): Loss of shape [b, s] if labels are provided, otherwise logits of shape [b, s, vocab_size].
        """
        use_inference_kv_cache = (
            inference_params is not None
            and "image_tokens_count" in inference_params.key_value_memory_dict
        )
        if use_inference_kv_cache:
            raise NotImplementedError()
        
        if self.pre_process:
            vision_embeds = None
            original_vision_grid_thw = vision_grid_thw.clone()
            vision_data, vision_grid_thw, seqlen_on_cp_ranks  = self._get_vision_cp_data(vision_data, vision_grid_thw)

            if vision_grid_thw.shape[0] > 0:
                vision_embeds = self.vision_model(
                    vision_data=vision_data, # If None, vision model should use intermediate outputs (EPP > 1)
                    grid_thw=vision_grid_thw # should provided in each EPP stage
                )

            if original_vision_grid_thw.shape[0] > 0 and self.language_model.config.context_parallel_size > 1 \
                and getattr(self.vision_model.config, 'enable_context_parallelism', False):
                if vision_embeds is None:
                    dtype = torch.float32
                    if self.vision_model.config.bf16:
                        dtype = torch.bfloat16
                    elif self.vision_model.config.fp16:
                        dtype = torch.float16
                    else:
                        print(f"WARN: vision model dtype is not bfloat16 or fp16, using float32")
                    vision_embeds = torch.zeros((0, self.language_model.config.hidden_size), device=vision_data.device, dtype=dtype)
                vision_embeds = AllGatherVisionEmbeddings.apply(vision_embeds, seqlen_on_cp_ranks)

            # If running inference, the language model KV cache will be updated for image token positions.
            # Here we store the image tokens sequence length, which can be used as an offset to the KV cache later.
            if inference_params is not None:
                raise NotImplementedError()
                # inference_params.key_value_memory_dict["image_tokens_count"] = (
                #     vision_embeddings.shape[0]
                # )
            
            # If running inference, we can skip image token computation if they were computed already earlier for this sample.
            if use_inference_kv_cache:
                language_embeddings: torch.Tensor = self.language_model.embedding(
                input_ids=input_ids,
                position_ids=None # NOTE: disable
                )  # [text_seq_len, b, h_language]
                # NOTE: why not cat here? is it the combined embeddings useless?
                combined_embeddings = language_embeddings
            elif vision_embeds is not None:
                if video_start_index == 0:
                    image_embeds = None
                    video_embeds = vision_embeds
                elif video_start_index == vision_embeds.shape[0]:
                    image_embeds = vision_embeds
                    video_embeds = None
                elif 0 < video_start_index < vision_embeds.shape[0]:
                    image_embeds = vision_embeds[:video_start_index]
                    video_embeds = vision_embeds[video_start_index:]
                else:
                    raise ValueError(f"Expect video token start index in range [0, {vision_embeds.shape[0]}], but got {video_start_index}")
                
                if image_embeds is not None:
                    image_input_mask = image_input_mask.T # shape [seqlen, mbs]
                if video_embeds is not None:
                    video_input_mask = video_input_mask.T
                # print(f"image_input_mask: {image_input_mask.sum()}")
                # print(f"video_input_mask: {video_input_mask.sum()}")
                combined_embeddings = self.language_model.embedding(
                    input_ids=input_ids,
                    position_ids=None, # NOTE: disable
                    image_input_mask=image_input_mask,
                    video_input_mask=video_input_mask,
                    image_embeds=image_embeds,
                    video_embeds=video_embeds
                )  # [text_seq_len, b, h_language]
            else:
                combined_embeddings = self.language_model.embedding(
                    input_ids=input_ids,
                    position_ids=None # NOTE: disable
                )  # [text_seq_len, b, h_language]
        else:
            combined_embeddings = None
        
        if self.context_parallel_lm > 1 or self.sequence_parallel_lm:
            combined_embeddings, labels, loss_mask, packed_seq_params = self._process_embedding_token_parallel(
                combined_embeddings, labels, loss_mask, packed_seq_params
            )

        # print(f"combined_embeddings.shape: {combined_embeddings.shape}")
        # print(f"attention_mask: {attention_mask}")
        # print(f"packed_seq_params: {packed_seq_params}")
        output = self.language_model(
            input_ids=None,
            position_ids=position_ids,              # None in encoder
            attention_mask=attention_mask,          # None in encoder
            decoder_input=combined_embeddings,      # only not None in the first decoder PP stage
            labels=labels,                          # only not None in the last decoder PP stage
            inference_params=inference_params,      # currently always None
            packed_seq_params=packed_seq_params,    # currently always None
            **(extra_block_kwargs or {}),
        )
        return output, loss_mask


def _load_state_dict_hook_ignore_param_names(
    param_names: List[str], module: torch.nn.Module, incompatible_keys: namedtuple
):
    """Hook to ignore missing keys during checkpoint loading.

    By default, this should not be used to avoid accidentally missing weights in checkpoint loading.

    Example use case: Use this for the vision projection if you want to load a checkpoint that contains vision and language model weights
    but not the vision projection weights.

    Args:
        param_names (list of str): Parameter names allowed to be missing when calling load_state_dict.
        module (torch.nn.Module): The torch module this hook applies to. Unused here but required by the torch API.
        incompatible_keys (namedtuple): Namedtuple with fields missing_keys and unexpected_keys, which collect the missing and unexpected
            keys when calling load_state_dict on this torch module, respectively.
    """
    for param_name in param_names:
        if param_name in incompatible_keys.missing_keys:
            logging.getLogger(__name__).warning(
                f"{param_name} being removed from incompatible_keys.missing_keys in QWen2VLModel"
            )
            incompatible_keys.missing_keys.remove(param_name)
