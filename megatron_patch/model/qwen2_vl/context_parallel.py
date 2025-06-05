import torch
from typing import Dict, Any
from megatron.core import parallel_state


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

class AllGatherLanguageEmbeddings(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, packed_seq_params):
        # input shape: [S/CP, B, H] or [S/CP, 1, H]
        cp_size = parallel_state.get_context_parallel_world_size()

        cp_rank = parallel_state.get_context_parallel_rank()
        outputs = [torch.empty_like(input) for _ in range(cp_size)]
        torch.distributed.all_gather(outputs, input, group=parallel_state.get_context_parallel_group())
        if packed_seq_params is None or packed_seq_params.format == 'sbhd':
            outputs = [x.chunk(2, dim=0) for x in outputs]
            reordered_outputs = []
            for i in range(cp_size*2):
                if i < cp_size:
                    reordered_outputs.append(outputs[i][0])
                else:
                    reordered_outputs.append(outputs[cp_size-i-1][1])
            # [S, x, H]
            output = torch.cat(reordered_outputs, dim=0)
        elif packed_seq_params.format == 'thd':
            pass

        ctx.cp_rank = cp_rank
        ctx.cp_size = cp_size
        ctx.packed_seq_params = packed_seq_params
        # ctx.save_for_backward(packed_seq_params)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        cp_size = ctx.cp_size

        cp_rank = ctx.cp_rank
        packed_seq_params = ctx.packed_seq_params
        if packed_seq_params is None or packed_seq_params.format == 'sbhd':
            grad_output = grad_output.view(2*cp_size, -1, *grad_output.shape[1:])
            index = torch.tensor(
                    [cp_rank, (2 * cp_size - cp_rank - 1)], device=grad_output.device)
            grad_output = grad_output.index_select(0, index)
            grad_output = grad_output.view(-1, *grad_output.shape[2:])

        elif packed_seq_params.format == 'thd':
            pass

        return grad_output, None

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