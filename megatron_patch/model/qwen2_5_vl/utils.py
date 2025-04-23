import torch
from typing import Dict, Any
from megatron.core import parallel_state


def get_batch_on_this_cp_rank(batch: Dict[str, Any]):
    """Slice batch input along sequence dimension into multiple chunks,
    which are parallelized across GPUs in a context parallel group.
    """

    # With causal masking, each token only attends to its prior tokens. Simply split
    # sequence into CP chunks can result in severe load imbalance. That's to say, chunks
    # at the end of sequence have bigger workload than others. To address this issue,
    # we split sequence into 2*CP ranks. Assuming CP=2, we then get 4 chunks, chunk_0
    # and chunk_3 are assigned to GPU0, chunk_1 and chunk_2 are assigned to GPU1, so
    # that we can get balanced workload among GPUs in a context parallel group.
    cp_size = parallel_state.get_context_parallel_world_size()
    if cp_size > 1:
        cp_rank = parallel_state.get_context_parallel_rank()
        for key, val in batch.items():
            if val is not None:
                #shape: [S, B, H], mask_shape: [B, 1, S]
                if key == 'combined_embeddings':
                    seq_dim = 0
                elif key == 'labels' or key == 'loss_mask':
                    seq_dim = 1
                else:
                    raise ValueError(f"Unsupported key: {key}")
                # seq_dim = 0 if key != 'attention_mask' else 2
                # print(f"key: {key}, val.shape: {val.shape}")
                val = val.view(
                    *val.shape[0:seq_dim],
                    2 * cp_size,
                    val.shape[seq_dim] // (2 * cp_size),
                    *val.shape[(seq_dim + 1) :],
                )
                index = torch.tensor(
                    [cp_rank, (2 * cp_size - cp_rank - 1)], device=val.device)
                val = val.index_select(seq_dim, index)
                val = val.view(*val.shape[0:seq_dim], -1, *val.shape[(seq_dim + 2) :])
                batch[key] = val

    return batch
