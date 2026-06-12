from __future__ import annotations

from typing import Any

from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerBase


def build_llama_model(
    model_config: dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
) -> LlamaForCausalLM:
    """Create a LLaMA causal language model from config values.

    Args:
        model_config: Model hyperparameters from the experiment config.
        tokenizer: Tokenizer used to set vocabulary and special-token ids.

    Returns:
        A randomly initialized LLaMA causal LM.
    """
    hidden_size = int(model_config.get("hidden_size", 256))
    num_attention_heads = int(model_config.get("num_attention_heads", 4))

    config = LlamaConfig(
        vocab_size=len(tokenizer),
        hidden_size=hidden_size,
        intermediate_size=int(model_config.get("intermediate_size", 4 * hidden_size)),
        num_hidden_layers=int(model_config.get("num_hidden_layers", 4)),
        num_attention_heads=num_attention_heads,
        num_key_value_heads=int(model_config.get("num_key_value_heads", num_attention_heads)),
        max_position_embeddings=int(model_config.get("context_length", 512)),
        rms_norm_eps=float(model_config.get("rms_norm_eps", 1e-5)),
        rope_theta=float(model_config.get("rope_theta", 10000.0)),
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    return LlamaForCausalLM(config)
