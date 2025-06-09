import torch
from typing import Dict, Any
from megatron.core import parallel_state
from megatron.core.packed_seq_params import PackedSeqParams
from torch.nn.functional import pad

try:
    import transformer_engine  # pylint: disable=unused-import

    from megatron.core.utils import is_te_min_version

    HAVE_TE = True
    try:
        import transformer_engine_torch as tex

        HAVE_TEX = True
    except:
        HAVE_TEX = False
except:
    HAVE_TE = False
    if parallel_state.get_context_parallel_world_size() > 1:
        raise RuntimeError("ContextParallelism requires TransformerEngine support, but not found.")

def sbhd_to_thd_format(input_tokens, position_ids, image_input_mask, video_input_mask, packed_seq_params: PackedSeqParams, pad_token_id):
    # input_tokens shape: [B, S]
    # position_ids shape: [3, B, S]
    # image_input_mask shape: [B, S]
    # video_input_mask shape: [B, S]
    cu_seqlen = packed_seq_params.cu_seqlens_q.cpu()
    cu_seqlen_padded = packed_seq_params.cu_seqlens_q_padded.cpu()
    bs = len(cu_seqlen) - 1
    seqlen = cu_seqlen[1:] - cu_seqlen[:-1]
    pad_seqlen = cu_seqlen_padded[1:] - cu_seqlen_padded[:-1]
    paddings = pad_seqlen - seqlen
    assert input_tokens.shape[0] == len(cu_seqlen) - 1, f"input_tokens.shape[0]={input_tokens.shape[0]}, len(cu_seqlen)={len(cu_seqlen)}"
    packed_tokens = []
    packed_position_ids = []
    packed_image_input_mask = []
    packed_video_input_mask = []

    for i in range(bs):
        padded_tokens = pad(input_tokens[i, :seqlen[i]], (0, paddings[i]), mode='constant', value=pad_token_id)
        packed_tokens.append(padded_tokens)

        padded_position_ids = pad(position_ids[:, i, :seqlen[i]], (0, paddings[i]), mode='constant', value=pad_token_id)
        packed_position_ids.append(padded_position_ids)

        if image_input_mask is not None:
            padded_image_input_mask = pad(image_input_mask[i, :seqlen[i]], (0, paddings[i]), mode='constant', value=False)
            packed_image_input_mask.append(padded_image_input_mask)
        if video_input_mask is not None:
            padded_video_input_mask = pad(video_input_mask[i, :seqlen[i]], (0, paddings[i]), mode='constant', value=False)
            packed_video_input_mask.append(padded_video_input_mask)

    packed_tokens = torch.cat(packed_tokens, dim=0)[None, :]
    packed_position_ids = torch.cat(packed_position_ids, dim=-1)[:,None,:]

    packed_image_input_mask = torch.cat(packed_image_input_mask, dim=0)[None, :] if image_input_mask is not None else None
    packed_video_input_mask = torch.cat(packed_video_input_mask, dim=0)[None, :] if video_input_mask is not None else None

    return packed_tokens, packed_position_ids, packed_image_input_mask, packed_video_input_mask

def thd_to_sbhd_format(tensor, packed_seq_params, max_pad_seqlen):
    # tensor shape: [S, 1, H]
    cu_seqlen = packed_seq_params.cu_seqlens_q.cpu()
    cu_seqlen_padded = packed_seq_params.cu_seqlens_q_padded.cpu()
    seqlen = cu_seqlen[1:] - cu_seqlen[:-1]
    # padded_seqlen = cu_seqlen_padded[1:] - cu_seqlen_padded[:-1]
    bs = len(cu_seqlen) - 1
    sbhd_tensor = torch.zeros((max_pad_seqlen, bs, tensor.shape[2]), device=tensor.device, dtype=tensor.dtype)
    for i in range(bs):
        start_idx = cu_seqlen_padded[i]
        end_idx = start_idx + seqlen[i]
        sbhd_tensor[:seqlen[i], i, :] = tensor[start_idx:end_idx, 0, :]
    return sbhd_tensor

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
        if packed_seq_params is None or packed_seq_params.qkv_format == 'sbhd':
            outputs = [x.chunk(2, dim=0) for x in outputs]
            reordered_outputs = []
            for i in range(cp_size*2):
                if i < cp_size:
                    reordered_outputs.append(outputs[i][0])
                else:
                    reordered_outputs.append(outputs[cp_size-i-1][1])
            # [S, x, H]
            output = torch.cat(reordered_outputs, dim=0)
        elif packed_seq_params.qkv_format == 'thd':
            cp_seqlen = input.shape[0]
            seqlen = cp_seqlen * cp_size
            ctx.seqlen = seqlen 
            output = torch.zeros((seqlen, *input.shape[1:]), device=input.device, dtype=input.dtype)
            for i in range(cp_size):
                index = tex.thd_get_partitioned_indices(
                    packed_seq_params.cu_seqlens_q_padded, seqlen, cp_size, i
                )
                output[index] = outputs[i]
                if i == cp_rank:
                    ctx.save_for_backward(index)

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
        if packed_seq_params is None or packed_seq_params.qkv_format == 'sbhd':
            grad_output = grad_output.view(2*cp_size, -1, *grad_output.shape[1:])
            index = torch.tensor(
                    [cp_rank, (2 * cp_size - cp_rank - 1)], device=grad_output.device)
            grad_output = grad_output.index_select(0, index)
            grad_output = grad_output.view(-1, *grad_output.shape[2:])

        elif packed_seq_params.qkv_format == 'thd':
            index = ctx.saved_tensors[0]
            grad_output = grad_output.index_select(0, index)

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

def get_pos_emb_on_this_cp_rank_thd(pos_emb, packed_seq_params):
    # pos_emb shape: [S, 1, H]
    cp_size = parallel_state.get_context_parallel_world_size()
    cp_rank = parallel_state.get_context_parallel_rank()
    seqlen = pos_emb.shape[0]
    index = tex.thd_get_partitioned_indices(
        packed_seq_params.cu_seqlens_q_padded, seqlen, cp_size, cp_rank)
    pos_emb_output = pos_emb.index_select(0, index)
    return pos_emb_output

class get_embeddings_on_this_cp_rank_thd(torch.autograd.Function):
    """Performs sharding for Context Parallelism in THD format

    In the forward pass, indices are selected for each CP rank and remaining tokens are dropped.
    In the backward pass, this class takes care of managing gradients for dropped tokens on each
    CP rank.
    """

    @staticmethod
    def forward(ctx, embeddings, packed_seq_params):
        """Context Parallelism forward support for THD format"""
        # embeddings shape: [S, 1, H], position_ids shape: [3, B, S]
        assert embeddings.shape[1] == 1, f"embeddings should be in THD format with shape [S, 1, H], but got {embeddings.shape}"

        assert HAVE_TEX and is_te_min_version(
                    "1.10.0"
                ), "Please update Transformer Engine to >= 1.10 to use \
                    Context Parallel with THD format data"
        cp_size = parallel_state.get_context_parallel_world_size()
        cp_rank = parallel_state.get_context_parallel_rank()
        seqlen = embeddings.shape[0]
        index = tex.thd_get_partitioned_indices(
            packed_seq_params.cu_seqlens_q_padded, seqlen, cp_size, cp_rank
        )
        ctx.save_for_backward(index)
        ctx.decoder_emb_seqlen = seqlen

        embed_output = embeddings.index_select(0, index)
        embed_output.requires_grad = embeddings.requires_grad

        return embed_output

    @staticmethod
    def backward(ctx, grad_out):
        """Context Parallelism backward support for THD format"""
        seqlen = ctx.decoder_emb_seqlen
        index = ctx.saved_tensors[0]
        assert grad_out.size(0) == index.size(
            0
        ), f"Shape mismatch in incoming gradient {grad_out.shape} and \
                index from THD CP sharding {index.shape}"
        grad_in = torch.zeros(
            seqlen,
            *grad_out.size()[1:],
            dtype=grad_out.dtype,
            device=grad_out.device,
        )
        grad_in[index] = grad_out

        return (grad_in, None)