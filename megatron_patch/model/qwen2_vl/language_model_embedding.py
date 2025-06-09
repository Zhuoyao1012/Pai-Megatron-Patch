# Copyright (c) 2024 Alibaba PAI and Nvidia Megatron-LM Team.
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

from typing import Literal

import torch
from torch import Tensor

from megatron.core import tensor_parallel
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.packed_seq_params import PackedSeqParams
from .context_parallel import get_batch_on_this_cp_rank, get_embeddings_on_this_cp_rank_thd


class LanguageModelEmbedding(MegatronModule):
    """Language model embeddings.

    Args:
        config (TransformerConfig): config object with all necessary configs for TransformerBlock
        vocab_size (int): vocabulary size
        max_sequence_length (int): maximum size of sequence. This
                             is used for positional embedding
        add_position_embedding (bool): Add a position embedding.
        embedding_dropout_prob (float): dropout probability for embeddings
        num_tokentypes (int): Set to 0 without binary head, and 2 with a binary head . Defaults to 0.
    """

    def __init__(
        self,
        config: TransformerConfig,
        vocab_size: int,
        max_sequence_length: int,
        position_embedding_type: Literal['learned_absolute', 'rope', 'none'] = 'learned_absolute',
        num_tokentypes: int = 0,
    ):
        super().__init__(config=config)

        self.config: TransformerConfig = config
        self.vocab_size: int = vocab_size
        self.max_sequence_length: int = max_sequence_length
        self.add_position_embedding: bool = position_embedding_type == 'learned_absolute'
        self.num_tokentypes = num_tokentypes
        self.reduce_scatter_embeddings = False

        # Word embeddings (parallel).
        self.word_embeddings = tensor_parallel.VocabParallelEmbedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.config.hidden_size,
            init_method=self.config.init_method,
            reduce_scatter_embeddings=self.reduce_scatter_embeddings,
            config=self.config,
        )

        # Position embedding (serial).
        if self.add_position_embedding:
            self.position_embeddings = torch.nn.Embedding(
                self.max_sequence_length, self.config.hidden_size
            )

            # Initialize the position embeddings.
            if self.config.perform_initialization:
                self.config.init_method(self.position_embeddings.weight)

        if self.num_tokentypes > 0:
            self.tokentype_embeddings = torch.nn.Embedding(
                self.num_tokentypes, self.config.hidden_size
            )
            # Initialize the token-type embeddings.
            if self.config.perform_initialization:
                self.config.init_method(self.tokentype_embeddings.weight)
        else:
            self.tokentype_embeddings = None

        # Embeddings dropout
        self.embedding_dropout = torch.nn.Dropout(self.config.hidden_dropout)

    def zero_parameters(self):
        """Zero out all parameters in embedding."""
        self.word_embeddings.weight.data.fill_(0)
        self.word_embeddings.weight.shared = True
        self.position_embeddings.weight.data.fill_(0)
        self.position_embeddings.weight.shared = True
        if self.num_tokentypes > 0:
            self.tokentype_embeddings.weight.data.fill_(0)
            self.tokentype_embeddings.weight.shared = True

    def _process_embedding_token_parallel(
        self, combined_embeddings, packed_seq_params
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

        """

        shard_factor = seq_dim = None
        if self.config.context_parallel_size > 1 and self.config.sequence_parallel:
            shard_factor = self.config.tensor_model_parallel_size * self.config.context_parallel_size * 2
            # seq_dim = 1
        elif self.config.context_parallel_size > 1:
            shard_factor = self.config.context_parallel_size * 2
            # seq_dim = 1
        elif self.config.sequence_parallel:
            shard_factor = self.config.tensor_model_parallel_size
            # seq_dim = 0

        seq_dim = 0
        assert (
            combined_embeddings.shape[seq_dim] % shard_factor == 0
        ), f"Sequence length should be divisible by {shard_factor} for \
            Sequence/Context parallelism"
        if self.config.sequence_parallel and self.config.tp_comm_overlap_lm:
            assert (
                combined_embeddings.shape[seq_dim] == self.config.max_sequence_length
            ), f"TP Comm overlap either requires Vision+Text token length \
            == language_max_sequence_length"

        if self.config.context_parallel_size > 1:
            # Distribute sequence across CP ranks
            if packed_seq_params is None or packed_seq_params.qkv_format == 'sbhd':
                batch = dict()
                batch["combined_embeddings"] = combined_embeddings
                batch = get_batch_on_this_cp_rank(batch)
                combined_embeddings = batch["combined_embeddings"]  # [S/CP, B, H]
            else:
                combined_embeddings = get_embeddings_on_this_cp_rank_thd.apply(combined_embeddings, packed_seq_params)
                # raise NotImplementedError("THD format data is not supported yet")


        return combined_embeddings

    def forward(
        self, 
        input_ids: Tensor, 
        position_ids: Tensor, 
        tokentype_ids: int = None,
        image_input_mask: Tensor = None,
        video_input_mask: Tensor = None,
        image_embeds: Tensor = None,
        video_embeds: Tensor = None, 
        packed_seq_params: PackedSeqParams = None,
    ) -> Tensor:
        """Forward pass of the embedding module.

        Args:
            input_ids (Tensor): The input tokens
            position_ids (Tensor): The position id's used to calculate position embeddings
            tokentype_ids (int): The token type ids. Used when args.bert_binary_head is set to True. Defaults to None

        Returns:
            Tensor: The output embeddings
        """
        word_embeddings = self.word_embeddings(input_ids)
        if self.add_position_embedding:
            position_embeddings = self.position_embeddings(position_ids)
            embeddings = word_embeddings + position_embeddings
        else:
            embeddings = word_embeddings

        if not self.reduce_scatter_embeddings:
            # Data format change to avoid explicit tranposes : [b s h] --> [s b h].
            embeddings = embeddings.transpose(0, 1).contiguous()

        if tokentype_ids is not None:
            assert self.tokentype_embeddings is not None
            # [b s h] -> [s b h] (So that it can be added with embeddings)
            tokentype_embedding = self.tokentype_embeddings(tokentype_ids).permute(1, 0, 2)
            embeddings = embeddings + tokentype_embedding
        else:
            assert self.tokentype_embeddings is None

        # If the input flag for fp32 residual connection is set, convert for float.
        if self.config.fp32_residual_connection:
            embeddings = embeddings.float()

        # Dropout.
        if self.config.sequence_parallel:
            if not self.reduce_scatter_embeddings:
                embeddings = embeddings.clone()
                if image_embeds is not None:
                    embeddings[image_input_mask] = image_embeds.to(embeddings.device, embeddings.dtype)
                if video_embeds is not None:
                    embeddings[video_input_mask] = video_embeds.to(embeddings.device, embeddings.dtype)

                embeddings = self._process_embedding_token_parallel(embeddings, packed_seq_params)
                embeddings = tensor_parallel.scatter_to_sequence_parallel_region(embeddings)
            # `scatter_to_sequence_parallel_region` returns a view, which prevents
            # the original tensor from being garbage collected. Clone to facilitate GC.
            # Has a small runtime cost (~0.5%).
            if self.config.clone_scatter_output_in_embedding:
                embeddings = embeddings.clone()
            with tensor_parallel.get_cuda_rng_tracker().fork():
                embeddings = self.embedding_dropout(embeddings)
        else:
            embeddings = embeddings.clone()
            if image_embeds is not None:
                embeddings[image_input_mask] = image_embeds.to(embeddings.device, embeddings.dtype)
            if video_embeds is not None:
                embeddings[video_input_mask] = video_embeds.to(embeddings.device, embeddings.dtype)
            embeddings = self._process_embedding_token_parallel(embeddings, packed_seq_params)
            embeddings = self.embedding_dropout(embeddings)

        return embeddings
