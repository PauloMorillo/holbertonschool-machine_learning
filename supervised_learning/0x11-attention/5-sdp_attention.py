#!/usr/bin/env python3
"""
This module has the method sdp_attention(Q, K, V, mask=None)
"""

import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension,
    i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on
    its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
      Q: query shape == (..., seq_len_q, depth)
      K: key shape == (..., seq_len_k, depth)
      V: value shape == (..., seq_len_v, depth_v)
      mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
      output, attention_weights
    """
    matmul_qk = tf.matmul(Q, K, transpose_b=True)
    # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    output = tf.matmul(attention_weights, V)  # (..., seq_len_q, depth_v)

    return output, attention_weights
