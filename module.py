import torch
import triton
import triton.language as tl
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions


def slice_linear_weights(linear_layer, active_indices, num_heads, head_dim):
    """
    Extract head-sliced linear weights for a subset of active heads.
    """
    weight = linear_layer.weight
    bias = linear_layer.bias

    w_view = weight.view(num_heads, head_dim, -1)
    b_view = bias.view(num_heads, head_dim)

    w_subset = torch.index_select(w_view, 0, active_indices)
    b_subset = torch.index_select(b_view, 0, active_indices)

    w_out = w_subset.reshape(-1, w_subset.size(-1))
    b_out = b_subset.reshape(-1)
    return w_out, b_out


def slice_qkv_weights(query_layer, key_layer, value_layer, active_indices, num_heads, head_dim):
    """
    Jointly slice Q/K/V head weights so the caller can run a single packed
    projection instead of three independent index_select + F.linear paths.
    """
    q_w, q_b = slice_linear_weights(query_layer, active_indices, num_heads, head_dim)
    k_w, k_b = slice_linear_weights(key_layer, active_indices, num_heads, head_dim)
    v_w, v_b = slice_linear_weights(value_layer, active_indices, num_heads, head_dim)

    packed_w = torch.cat([q_w, k_w, v_w], dim=0).contiguous()
    packed_b = torch.cat([q_b, k_b, v_b], dim=0).contiguous()
    return packed_w, packed_b


@triton.jit
def _compact_rows_kernel(
    in_ptr,
    index_ptr,
    out_ptr,
    stride_in_row,
    stride_in_col,
    stride_out_row,
    stride_out_col,
    keep_tokens,
    width,
    BLOCK_T: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_w = tl.program_id(1)

    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_w = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)

    token_mask = offs_t < keep_tokens
    width_mask = offs_w < width
    src_rows = tl.load(index_ptr + offs_t, mask=token_mask, other=0).to(tl.int32)

    in_ptrs = in_ptr + src_rows[:, None] * stride_in_row + offs_w[None, :] * stride_in_col
    out_ptrs = out_ptr + offs_t[:, None] * stride_out_row + offs_w[None, :] * stride_out_col
    mask = token_mask[:, None] & width_mask[None, :]
    values = tl.load(in_ptrs, mask=mask, other=0.0)
    tl.store(out_ptrs, values, mask=mask)


@triton.jit
def _compact_vector_kernel(
    in_ptr,
    index_ptr,
    out_ptr,
    keep_tokens,
    BLOCK_T: tl.constexpr,
):
    pid_t = tl.program_id(0)
    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    token_mask = offs_t < keep_tokens
    src_rows = tl.load(index_ptr + offs_t, mask=token_mask, other=0).to(tl.int32)
    values = tl.load(in_ptr + src_rows, mask=token_mask, other=0.0)
    tl.store(out_ptr + offs_t, values, mask=token_mask)


def _triton_compact_hidden_states_batch1(hidden_states, active_token_indices):
    keep_tokens = active_token_indices.size(1)
    hidden_size = hidden_states.size(-1)
    token_idx = active_token_indices[0].contiguous().to(torch.int32)
    source = hidden_states[0].contiguous()
    compacted = torch.empty((keep_tokens, hidden_size), device=hidden_states.device, dtype=hidden_states.dtype)

    grid = (
        triton.cdiv(keep_tokens, 64),
        triton.cdiv(hidden_size, 128),
    )
    _compact_rows_kernel[grid](
        source,
        token_idx,
        compacted,
        source.stride(0),
        source.stride(1),
        compacted.stride(0),
        compacted.stride(1),
        keep_tokens,
        hidden_size,
        BLOCK_T=64,
        BLOCK_W=128,
        num_warps=4,
        num_stages=1,
    )
    return compacted.unsqueeze(0)


def _triton_compact_vector_batch1(vector_2d, active_token_indices):
    keep_tokens = active_token_indices.size(1)
    token_idx = active_token_indices[0].contiguous().to(torch.int32)
    source = vector_2d[0].contiguous()
    compacted = torch.empty((keep_tokens,), device=vector_2d.device, dtype=vector_2d.dtype)

    grid = (triton.cdiv(keep_tokens, 256),)
    _compact_vector_kernel[grid](
        source,
        token_idx,
        compacted,
        keep_tokens,
        BLOCK_T=256,
        num_warps=4,
        num_stages=1,
    )
    return compacted.unsqueeze(0)


def _triton_compact_attention_mask_batch1(attention_mask, active_token_indices):
    keep_tokens = active_token_indices.size(1)
    token_idx = active_token_indices[0].contiguous().to(torch.int32)
    source = attention_mask[0, 0, 0].contiguous()
    compacted = torch.empty((keep_tokens,), device=attention_mask.device, dtype=attention_mask.dtype)

    grid = (triton.cdiv(keep_tokens, 256),)
    _compact_vector_kernel[grid](
        source,
        token_idx,
        compacted,
        keep_tokens,
        BLOCK_T=256,
        num_warps=4,
        num_stages=1,
    )
    return compacted.view(1, 1, 1, keep_tokens)


def compact_token_state(
    hidden_states,
    active_token_indices,
    attention_mask=None,
    cumulative_token_score=None,
):
    if (
        hidden_states.is_cuda
        and active_token_indices.is_cuda
        and hidden_states.size(0) == 1
        and active_token_indices.size(0) == 1
    ):
        hidden_states = _triton_compact_hidden_states_batch1(hidden_states, active_token_indices)
        if attention_mask is not None and attention_mask.is_cuda and attention_mask.size(0) == 1:
            attention_mask = _triton_compact_attention_mask_batch1(attention_mask, active_token_indices)
        elif attention_mask is not None:
            attention_mask = attention_mask.index_select(3, active_token_indices[0])

        if cumulative_token_score is not None and cumulative_token_score.is_cuda and cumulative_token_score.size(0) == 1:
            cumulative_token_score = _triton_compact_vector_batch1(cumulative_token_score, active_token_indices)
        elif cumulative_token_score is not None:
            cumulative_token_score = cumulative_token_score.index_select(1, active_token_indices[0])
        return hidden_states, attention_mask, cumulative_token_score

    expanded_indices = active_token_indices.unsqueeze(-1).expand(-1, -1, hidden_states.size(-1))
    hidden_states = torch.gather(hidden_states, dim=1, index=expanded_indices)

    if attention_mask is not None:
        mask_indices = active_token_indices.unsqueeze(1).unsqueeze(2)
        attention_mask = torch.gather(attention_mask, 3, index=mask_indices)

    if cumulative_token_score is not None:
        cumulative_token_score = torch.gather(cumulative_token_score, 1, active_token_indices)

    return hidden_states, attention_mask, cumulative_token_score


def should_compact_token_state(hidden_states, active_token_indices, attn_module, steps_since_prune):
    if active_token_indices is None:
        return False

    if not getattr(attn_module, "enable_delayed_token_compaction", False):
        return True

    compact_interval = max(1, int(getattr(attn_module, "token_compact_interval", 1)))
    min_drop_ratio = float(getattr(attn_module, "token_compact_min_drop_ratio", 1.0))

    current_tokens = hidden_states.size(1)
    keep_tokens = active_token_indices.size(1)
    if current_tokens <= 0:
        return True

    drop_ratio = 1.0 - (keep_tokens / current_tokens)
    if drop_ratio >= min_drop_ratio:
        return True

    return steps_since_prune >= compact_interval


def spatten_encoder_forward(self, hidden_states, attention_mask=None, **kwargs):
    active_head_indices = None
    active_token_indices = None
    cumulative_token_score = None
    pending_token_source_layer = None

    for layer_idx, layer_module in enumerate(self.layer):
        if active_token_indices is not None:
            steps_since_prune = layer_idx - pending_token_source_layer
            if should_compact_token_state(
                hidden_states,
                active_token_indices,
                layer_module.attention.self,
                steps_since_prune,
            ):
                hidden_states, attention_mask, cumulative_token_score = compact_token_state(
                    hidden_states,
                    active_token_indices,
                    attention_mask=attention_mask,
                    cumulative_token_score=cumulative_token_score,
                )
                active_token_indices = None
                pending_token_source_layer = None

        layer_module.attention.self.active_head_indices_for_this_layer = active_head_indices
        layer_module.attention.self.cumulative_token_score = cumulative_token_score

        kwargs.pop("use_cache", None)
        kwargs.pop("output_hidden_states", None)
        kwargs.pop("return_dict", None)
        layer_outputs = layer_module(
            hidden_states,
            attention_mask=attention_mask,
            **kwargs
        )

        if isinstance(layer_outputs, (tuple, list)):
            hidden_states = layer_outputs[0]
        else:
            hidden_states = layer_outputs

        active_head_indices = layer_module.attention.self.next_active_head_indices
        active_token_indices = layer_module.attention.self.next_active_token_indices
        cumulative_token_score = layer_module.attention.self.cumulative_token_score
        if active_token_indices is not None:
            pending_token_source_layer = layer_idx

    if active_token_indices is not None:
        hidden_states, attention_mask, cumulative_token_score = compact_token_state(
            hidden_states,
            active_token_indices,
            attention_mask=attention_mask,
            cumulative_token_score=cumulative_token_score,
        )

    return BaseModelOutputWithPastAndCrossAttentions(
        last_hidden_state=hidden_states,
        past_key_values=None,
        hidden_states=None,
        attentions=None,
        cross_attentions=None,
    )
