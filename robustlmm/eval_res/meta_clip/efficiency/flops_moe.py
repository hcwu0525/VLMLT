"""
Inference FLOPs calculation

Borrowed from nanotron: https://github.com/huggingface/nanotron/blob/cfcdeae812d4c7210990932ebcce5f672084408d/src/nanotron/models/llama.py#L1100
"""
def convert_to_human_readable_size(num):
    if num / 1e9 > 1:
        return f"{num / 1e9:.2f} B"
    elif num / 1e6 > 1:
        return f"{num / 1e6:.2f} M"
    elif num / 1e3 > 1:
        return f"{num / 1e3:.2f} K"
    else:
        return f"{num}"


def get_flops(
    num_layers,
    hidden_size,
    num_heads,
    num_key_value_heads,
    vocab_size,
    seq_len,
    ffn_hidden_size,
    batch_size=1,
):
    """Counts flops in an decoder-only model
    Args:
        num_layers: number of decoder layers
        hidden_size: hidden size of the model
        num_heads: number of heads in the model
        num_key_value_heads: number of key/value heads in the model
        ffn_hidden_size: hidden size of the FFN
        vocab_size: size of the vocabulary
        seq_len: sequence length of the decoder
        batch_size: batch size
    Returns:
        model_flops: flops in the model (should be independent of the hardware and model implementation)
        hardware_flops: flops in the hardware (actual flops performed on the hardware). Check 6.3 in https://arxiv.org/pdf/2205.05198.pdf

    Ref: copied from nanotron
    """
    if num_key_value_heads is None:
        num_key_value_heads = num_heads
    hidden_size_per_head = hidden_size // num_heads
    # In the following we mark the reduced dimension with parentheses
    # decoder
    # self attention
    ## qkv projection
    decoder_qkv_proj_flops_fwd = (
        2
        * num_layers
        * batch_size
        * seq_len
        * (hidden_size)
        * num_heads
        * hidden_size_per_head
        + 2
        * num_layers
        * batch_size
        * seq_len
        * (hidden_size)
        * 2
        * num_key_value_heads
        * hidden_size_per_head
    )
    ## qk logits
    decoder_qk_logits_flops_fwd = (
        2
        * num_layers
        * batch_size
        * num_heads
        * seq_len
        * (hidden_size_per_head)
        * seq_len
    )
    ## v logits
    decoder_v_logits_flops_fwd = (
        2
        * num_layers
        * batch_size
        * num_heads
        * seq_len
        * (seq_len)
        * hidden_size_per_head
    )
    ## attn out
    decoder_attn_out_flops_fwd = (
        2
        * num_layers
        * batch_size
        * num_heads
        * seq_len
        * (hidden_size_per_head)
        * hidden_size
    )
    # FF
    ## 1st layer
    decoder_ffn_1_flops_fwd = (
        4 * num_layers * batch_size * seq_len * (hidden_size) * ffn_hidden_size
    )
    ## 2nd layer
    decoder_ffn_2_flops_fwd = (
        2 * num_layers * batch_size * seq_len * (ffn_hidden_size) * hidden_size
    )

    decoder_flops_fwd = (
        decoder_qkv_proj_flops_fwd
        + decoder_qk_logits_flops_fwd
        + decoder_v_logits_flops_fwd
        + decoder_attn_out_flops_fwd
        + decoder_ffn_1_flops_fwd
        + decoder_ffn_2_flops_fwd
    )

    # lm head
    lm_head_flops_fwd = 2 * batch_size * seq_len * (hidden_size) * vocab_size

    # the bwd pass requires double the flops in case of matmuls to calculate the gradients with respect to
    # both input and weight tensors
    model_flops = 1 * (decoder_flops_fwd + lm_head_flops_fwd)  # 1 for fwd
    # model_flops = 3 * (decoder_flops_fwd + lm_head_flops_fwd)  # 1 for fwd + 2 for bwd

    hardware_flops = model_flops  # TODO: This is a placeholder for now

    attn_flops = (
        decoder_qkv_proj_flops_fwd
        + decoder_qk_logits_flops_fwd
        + decoder_v_logits_flops_fwd
        + decoder_attn_out_flops_fwd
    )
    ffn_flops = decoder_ffn_1_flops_fwd + decoder_ffn_2_flops_fwd

    # print(f"attn: {attn_flops / model_flops * 100:.2f}%")
    # print(f"ffn: {ffn_flops / model_flops * 100:.2f}%")
    # print(f"attn / ffn: {attn_flops / ffn_flops:.2f}")

    return model_flops, hardware_flops, attn_flops, ffn_flops, attn_flops / ffn_flops


def get_moe_flops(
    num_layers,
    hidden_size,
    num_heads,
    num_key_value_heads,
    vocab_size,
    seq_len,
    ffn_hidden_size,
    topk: int,
    experts: int,
    batch_size=1,
):
    """Counts flops in an decoder-only model
    Args:
        num_layers: number of decoder layers
        hidden_size: hidden size of the model
        num_heads: number of heads in the model
        num_key_value_heads: number of key/value heads in the model
        ffn_hidden_size: hidden size of the FFN
        vocab_size: size of the vocabulary
        seq_len: sequence length of the decoder
        batch_size: batch size
    Returns:
        model_flops: flops in the model (should be independent of the hardware and model implementation)
        hardware_flops: flops in the hardware (actual flops performed on the hardware). Check 6.3 in https://arxiv.org/pdf/2205.05198.pdf

    Ref: copied from nanotron
    """
    if num_key_value_heads is None:
        num_key_value_heads = num_heads
    hidden_size_per_head = hidden_size // num_heads
    # In the following we mark the reduced dimension with parentheses
    # decoder
    # self attention
    ## qkv projection
    decoder_qkv_proj_flops_fwd = (
        2
        * num_layers
        * batch_size
        * seq_len
        * (hidden_size)
        * num_heads
        * hidden_size_per_head
        + 2
        * num_layers
        * batch_size
        * seq_len
        * (hidden_size)
        * 2
        * num_key_value_heads
        * hidden_size_per_head
    )
    ## qk logits
    decoder_qk_logits_flops_fwd = (
        2
        * num_layers
        * batch_size
        * num_heads
        * seq_len
        * (hidden_size_per_head)
        * seq_len
    )
    ## v logits
    decoder_v_logits_flops_fwd = (
        2
        * num_layers
        * batch_size
        * num_heads
        * seq_len
        * (seq_len)
        * hidden_size_per_head
    )
    ## attn out
    decoder_attn_out_flops_fwd = (
        2
        * num_layers
        * batch_size
        * num_heads
        * seq_len
        * (hidden_size_per_head)
        * hidden_size
    )
    # FF
    ## 1st layer
    decoder_ffn_1_flops_fwd = (
        4 * num_layers * batch_size * seq_len * (hidden_size) * (topk * ffn_hidden_size)
    )
    ## 2nd layer
    decoder_ffn_2_flops_fwd = (
        2 * num_layers * batch_size * seq_len * (topk * ffn_hidden_size) * hidden_size
    )
    decoder_gate_flops_fwd = (
        2 * num_layers * batch_size * seq_len * hidden_size * experts
    )

    decoder_flops_fwd = (
        decoder_qkv_proj_flops_fwd
        + decoder_qk_logits_flops_fwd
        + decoder_v_logits_flops_fwd
        + decoder_attn_out_flops_fwd
        + decoder_ffn_1_flops_fwd
        + decoder_ffn_2_flops_fwd
        + decoder_gate_flops_fwd
    )

    # lm head
    lm_head_flops_fwd = 2 * batch_size * seq_len * (hidden_size) * vocab_size

    # the bwd pass requires double the flops in case of matmuls to calculate the gradients with respect to
    # both input and weight tensors
    model_flops = 1 * (decoder_flops_fwd + lm_head_flops_fwd)  # 1 for fwd
    # model_flops = 3 * (decoder_flops_fwd + lm_head_flops_fwd)  # 1 for fwd + 2 for bwd

    hardware_flops = model_flops  # TODO: This is a placeholder for now

    attn_flops = (
        decoder_qkv_proj_flops_fwd
        + decoder_qk_logits_flops_fwd
        + decoder_v_logits_flops_fwd
        + decoder_attn_out_flops_fwd
    )
    ffn_flops = (
        decoder_ffn_1_flops_fwd
        + decoder_ffn_2_flops_fwd
        + decoder_gate_flops_fwd
    )

    return model_flops, hardware_flops, attn_flops, ffn_flops, attn_flops / ffn_flops


def get_moa_moe_flops(
    num_layers,
    hidden_size,
    num_heads,
    num_key_value_heads,
    vocab_size,
    seq_len,
    ffn_hidden_size,
    attn_topk: int,
    attn_experts: int,
    ffn_topk: int,
    ffn_experts: int,
    batch_size=1,
):
    """Counts flops in an decoder-only model
    Args:
        num_layers: number of decoder layers
        hidden_size: hidden size of the model
        num_heads: number of heads in the model
        num_key_value_heads: number of key/value heads in the model
        ffn_hidden_size: hidden size of the FFN
        vocab_size: size of the vocabulary
        seq_len: sequence length of the decoder
        batch_size: batch size
    Returns:
        model_flops: flops in the model (should be independent of the hardware and model implementation)
        hardware_flops: flops in the hardware (actual flops performed on the hardware). Check 6.3 in https://arxiv.org/pdf/2205.05198.pdf

    Ref: copied from nanotron
    """
    if num_key_value_heads is None:
        num_key_value_heads = num_heads
    assert attn_topk <= attn_experts <= num_heads
    hidden_size_per_head = hidden_size // num_heads
    attn_act_size = attn_topk * hidden_size_per_head
    # In the following we mark the reduced dimension with parentheses
    # decoder
    # self attention
    ## qkv projection
    decoder_qkv_proj_flops_fwd = (
        2 * num_layers * batch_size * seq_len * (hidden_size) * attn_act_size
        + 2
        * num_layers
        * batch_size
        * seq_len
        * (hidden_size)
        * 2
        * num_key_value_heads
        * hidden_size_per_head
    )
    ## qk logits
    decoder_qk_logits_flops_fwd = (
        2 * num_layers * batch_size * attn_act_size * seq_len * seq_len
    )
    ## v logits
    decoder_v_logits_flops_fwd = (
        2 * num_layers * batch_size * seq_len * (seq_len) * attn_act_size
    )
    ## attn out
    decoder_attn_out_flops_fwd = (
        2 * num_layers * batch_size * seq_len * hidden_size * attn_act_size
    )
    # attn gate B x hsz -> B x attn_experts
    decoder_attn_gate_flops_fwd = (
        2 * num_layers * batch_size * seq_len * hidden_size * attn_experts
    )
    # FF
    ## 1st layer
    decoder_ffn_1_flops_fwd = (
        4
        * num_layers
        * batch_size
        * seq_len
        * (hidden_size)
        * (ffn_topk * ffn_hidden_size)
    )
    ## 2nd layer
    decoder_ffn_2_flops_fwd = (
        2
        * num_layers
        * batch_size
        * seq_len
        * (ffn_topk * ffn_hidden_size)
        * hidden_size
    )
    decoder_gate_flops_fwd = (
        2 * num_layers * batch_size * seq_len * hidden_size * ffn_experts
        + 2 * num_layers * batch_size * seq_len * ffn_experts * ffn_experts
    )

    decoder_flops_fwd = (
        decoder_qkv_proj_flops_fwd
        + decoder_qk_logits_flops_fwd
        + decoder_v_logits_flops_fwd
        + decoder_attn_out_flops_fwd
        + decoder_attn_gate_flops_fwd
        + decoder_ffn_1_flops_fwd
        + decoder_ffn_2_flops_fwd
        + decoder_gate_flops_fwd
    )

    # lm head
    lm_head_flops_fwd = 2 * batch_size * seq_len * (hidden_size) * vocab_size

    # the bwd pass requires double the flops in case of matmuls to calculate the gradients with respect to
    # both input and weight tensors
    model_flops = 1 * (decoder_flops_fwd + lm_head_flops_fwd)  # 1 for fwd
    # model_flops = 3 * (decoder_flops_fwd + lm_head_flops_fwd)  # 1 for fwd + 2 for bwd

    hardware_flops = model_flops  # TODO: This is a placeholder for now

    attn_flops = (
        decoder_qkv_proj_flops_fwd
        + decoder_qk_logits_flops_fwd
        + decoder_v_logits_flops_fwd
        + decoder_attn_out_flops_fwd
        + decoder_attn_gate_flops_fwd
    )

    ffn_flops = (
        decoder_ffn_1_flops_fwd
        + decoder_ffn_2_flops_fwd
        + decoder_gate_flops_fwd
    )

    return model_flops, hardware_flops, attn_flops, ffn_flops, attn_flops / ffn_flops


def get_moa_flops(
    num_layers,
    hidden_size,
    num_heads,
    num_key_value_heads,
    vocab_size,
    seq_len,
    ffn_hidden_size,
    attn_topk: int,
    attn_experts: int,
    batch_size=1,
):
    """Counts flops in an decoder-only model
    Args:
        num_layers: number of decoder layers
        hidden_size: hidden size of the model
        num_heads: number of heads in the model
        num_key_value_heads: number of key/value heads in the model
        ffn_hidden_size: hidden size of the FFN
        vocab_size: size of the vocabulary
        seq_len: sequence length of the decoder
        batch_size: batch size
    Returns:
        model_flops: flops in the model (should be independent of the hardware and model implementation)
        hardware_flops: flops in the hardware (actual flops performed on the hardware). Check 6.3 in https://arxiv.org/pdf/2205.05198.pdf

    Ref: copied from nanotron
    """
    if num_key_value_heads is None:
        num_key_value_heads = num_heads
    assert attn_topk <= attn_experts <= num_heads
    hidden_size_per_head = hidden_size // num_heads
    attn_act_size = attn_topk * (hidden_size / attn_experts)
    # In the following we mark the reduced dimension with parentheses
    # decoder
    # self attention
    ## qkv projection
    decoder_qkv_proj_flops_fwd = (
        2 * num_layers * batch_size * seq_len * (hidden_size) * attn_act_size
        + 2
        * num_layers
        * batch_size
        * seq_len
        * (hidden_size)
        * 2
        * num_key_value_heads
        * hidden_size_per_head
    )
    ## qk logits
    decoder_qk_logits_flops_fwd = (
        2 * num_layers * batch_size * attn_act_size * seq_len * seq_len
    )
    ## v logits
    decoder_v_logits_flops_fwd = (
        2 * num_layers * batch_size * seq_len * (seq_len) * attn_act_size
    )
    ## attn out
    decoder_attn_out_flops_fwd = (
        2 * num_layers * batch_size * seq_len * hidden_size * attn_act_size
    )
    # attn gate B x hsz -> B x attn_experts
    decoder_attn_gate_flops_fwd = (
        2 * num_layers * batch_size * seq_len * hidden_size * attn_experts
    )
    # FF
    ## 1st layer
    decoder_ffn_1_flops_fwd = (
        4 * num_layers * batch_size * seq_len * (hidden_size) * (ffn_hidden_size)
    )
    ## 2nd layer
    decoder_ffn_2_flops_fwd = (
        2 * num_layers * batch_size * seq_len * (ffn_hidden_size) * hidden_size
    )
    decoder_flops_fwd = (
        decoder_qkv_proj_flops_fwd
        + decoder_qk_logits_flops_fwd
        + decoder_v_logits_flops_fwd
        + decoder_attn_out_flops_fwd
        + decoder_attn_gate_flops_fwd
        + decoder_ffn_1_flops_fwd
        + decoder_ffn_2_flops_fwd
    )

    # lm head
    lm_head_flops_fwd = 2 * batch_size * seq_len * (hidden_size) * vocab_size

    # the bwd pass requires double the flops in case of matmuls to calculate the gradients with respect to
    # both input and weight tensors
    model_flops = 1 * (decoder_flops_fwd + lm_head_flops_fwd)  # 1 for fwd
    # model_flops = 3 * (decoder_flops_fwd + lm_head_flops_fwd)  # 1 for fwd + 2 for bwd

    hardware_flops = model_flops  # TODO: This is a placeholder for now

    attn_flops = (
        decoder_qkv_proj_flops_fwd
        + decoder_qk_logits_flops_fwd
        + decoder_v_logits_flops_fwd
        + decoder_attn_out_flops_fwd
        + decoder_attn_gate_flops_fwd
    )
    ffn_flops = (
        decoder_ffn_1_flops_fwd
        + decoder_ffn_2_flops_fwd
    )

    return model_flops, hardware_flops, attn_flops, ffn_flops, attn_flops / ffn_flops


def _flops_calc():
    # res = get_flops(32, 4096, 32, 32, 32000, 4096, 11008)
    # res = res[0] / 1e12
    # print(f"llama2: {res:,.1f} TFLOPs")
    # res = get_moe_flops(32, 4096, 32, 32, 32000, 4096, 11008 / 16, 2, 16)
    # res = res[0] / 1e12
    # print(f"llama-moe 3.0B 2/16: {res:,.1f} TFLOPs")
    # res = get_moe_flops(32, 4096, 32, 32, 32000, 4096, 11008 / 16, 4, 16)
    # res = res[0] / 1e12
    # print(f"llama-moe 3.5B 4/16: {res:,.1f} TFLOPs")
    # res = get_moe_flops(32, 4096, 32, 32, 32000, 4096, 11008 / 8, 2, 8)
    # res = res[0] / 1e12
    # print(f"llama-moe 3.5B 2/8: {res:,.1f} TFLOPs")

    # llama3 8B
    res = get_flops(32, 4096, 32, 8, 128256, 4096, 14336, 8)
    res = res[0] / 1e12
    print(f"llama3 8B: {res:,.1f} TFLOPs")

    # llama3 8B FFN MoE 2/8E
    res = get_moe_flops(32, 4096, 32, 8, 128256, 4096, 14336 / 8, 2, 8, 8)
    res = res[0] / 1e12
    print(f"llama3 8B FFN 2/8E: {res:,.1f} TFLOPs")

    # llama3 8B FFN MoE 4/8E
    res = get_moe_flops(32, 4096, 32, 8, 128256, 4096, 14336 / 8, 4, 8, 8)
    res = res[0] / 1e12
    print(f"llama3 8B FFN 4/8E: {res:,.1f} TFLOPs")

    # llama3 8B FFN MoE 8/8E
    res = get_moe_flops(32, 4096, 32, 8, 128256, 4096, 14336 / 8, 8, 8, 8)
    res = res[0] / 1e12
    print(f"llama3 8B FFN 8/8E: {res:,.1f} TFLOPs")

    # llama3 8B Attention MoE 2/8E
    res = get_moa_flops(32, 4096, 32, 8, 128256, 4096, 14336, 2, 8, 8)
    res = res[0] / 1e12
    print(f"llama3 8B Attention 2/8E: {res:,.1f} TFLOPs")

    # llama3 8B Attention MoE 4/8E
    res = get_moa_flops(32, 4096, 32, 8, 128256, 4096, 14336, 4, 8, 8)
    res = res[0] / 1e12
    print(f"llama3 8B Attention 4/8E: {res:,.1f} TFLOPs")

    # llama3 8B Attention MoE 6/8E
    res = get_moa_flops(32, 4096, 32, 8, 128256, 4096, 14336, 6, 8, 8)
    res = res[0] / 1e12
    print(f"llama3 8B Attention 6/8E: {res:,.1f} TFLOPs")

    # llama3 8B Attention MoE 8/8E
    res = get_moa_flops(32, 4096, 32, 8, 128256, 4096, 14336, 8, 8, 8)
    res = res[0] / 1e12
    print(f"llama3 8B Attention 8/8E: {res:,.1f} TFLOPs")

    print("=" * 100 + "\n128k")
    # llama3 8B
    res = get_flops(32, 4096, 32, 8, 128256, 128000, 14336, 8)
    res = res[0] / 1e12
    print(f"llama3 8B: {res:,.1f} TFLOPs")

    # llama3 8B FFN MoE 2/8E
    res = get_moe_flops(32, 4096, 32, 8, 128256, 128000, 14336 / 8, 2, 8, 8)
    res = res[0] / 1e12
    print(f"llama3 8B FFN 2/8E: {res:,.1f} TFLOPs")

    # llama3 8B Attention MoE 2/8E
    res = get_moa_flops(32, 4096, 32, 8, 128256, 128000, 14336, 2, 8, 8)
    res = res[0] / 1e12
    print(f"llama3 8B Attention 2/8E: {res:,.1f} TFLOPs")


def context_len_flops_ratio():
    import matplotlib.pyplot as plt
    # context len
    context_lens = []
    attn_flops_ratios = []
    ffn_flops_ratios = []
    for i in range(10, 21, 1):
        res = get_flops(32, 4096, 32, 8, 128256, 2**i, 14336, 1)
        context_lens.append(2**i)
        attn_flops_ratios.append(res[2] / res[0])
        ffn_flops_ratios.append(res[3] / res[0])
        print(f"len={2**i}, attn ratio: {res[2] / res[0]:.2f}, ffn ratio: {res[3] / res[0]:.2f}")

    plt.plot(context_lens, attn_flops_ratios, label="Attention")
    plt.plot(context_lens, ffn_flops_ratios, label="FFN")
    plt.legend()
    plt.xlabel("Context Length")
    plt.ylabel("FLOPs Ratio")
    plt.grid(zorder=-1)
    plt.show()


def context_len_methods_comparison():
    import matplotlib.pyplot as plt
    context_lens = []
    base_flops = []
    moe_flops = []
    moa_flops = []
    # for i in range(1, 16, 1):
    for i in range(10, 21, 1):
        seq_len = 2**i
        context_lens.append(seq_len)
        res = get_flops(32, 4096, 32, 8, 128256, seq_len, 14336, 8)
        base_flops.append(res[0])
        res = get_moe_flops(32, 4096, 32, 8, 128256, seq_len, 14336 / 8, 2, 8, 8)
        moe_flops.append(res[0])
        res = get_moa_flops(32, 4096, 32, 8, 128256, seq_len, 14336, 2, 8, 8)
        moa_flops.append(res[0])
    
    plt.plot(context_lens, base_flops, label="Baseline")
    plt.plot(context_lens, moe_flops, label="MoE")
    plt.plot(context_lens, moa_flops, label="MoA")
    plt.legend()
    plt.xlabel("Context Length")
    plt.ylabel("FLOPs")
    plt.grid(zorder=-1)
    plt.show()



if __name__ == "__main__":
    # _flops_calc()
    context_len_methods_comparison()
