from __future__ import absolute_import, division, print_function, unicode_literals


import numpy as np

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers, backend

import param as P


tf.keras.backend.clear_session()  # For easy reset of notebook state.


class n_gram_layer(layers.Layer):
    def __init__(self):
        super(n_gram_layer, self).__init__()

    def call(self, inputs, **kwargs):
        stack_tensor = tf.stack(inputs, 1)
        stack_tensor = tf.reverse(stack_tensor)
        index = tf.constant(len(stack_tensor))
        expect_result = tf.zeros([P.batch_size * P.attr_num * P.instance_num, P.char_embed_size])

        def condition(idx, summation):
            return tf.greater(idx, 0)

        def body(idx, summation):
            precess = tf.slice(stack_tensor, [0, idx - 1, 0], [-1, -1, -1])
            summand = tf.reduce_sum(precess, 1)
            return tf.subtract(idx, 1), tf.add(summand, summation)

        result = tf.while_loop(condition, body, [index, expect_result])
        return result[1]


class self_attention_layer(layers.Layer):
    def __init__(self, units):
        self.units = units
        super(self_attention_layer, self).__init__()

    def call(self, inputs, **kwargs):
        attention_context_vector = self.add_weight("attention context vector", [P.attention_size], dtype=tf.float32,
                                                   regularizer=keras.regularizers.l2(1e-4))
        input_projection = layers.Dense(P.attention_size, activation='relu')(inputs)
        vector_attn = tf.reduce_sum(tf.multiply(input_projection, attention_context_vector), axis=2,
                                    keep_dims=True)
        attention_weights = tf.nn.softmax(vector_attn, dim=1)
        weighted_projection = tf.multiply(inputs, attention_weights)
        outputs = tf.reduce_sum(weighted_projection, axis=1)
        return outputs


"""
    default input title shape = (batch_size, P.title_len)
    default input attr shape = (batch_size, P.attr_num * P.attr_len)
    default input instance shape = (batch_size, P.attr_num * P.ins_num * P.ins_len)
"""
title_input = keras.Input(shape=(P.title_len,), name='table_name')
attr_input = keras.Input(shape=(P.attr_num, P.attr_len,), name='attr_name')

embedding_layer = layers.Embedding(P.char_num, P.char_embed_size, mask_zero=True)

title_features = embedding_layer(title_input)
attr_features = embedding_layer(attr_input)

title_features = layers.LSTM(P.lstm_units)(title_features)
attr_features = layers.LSTM(P.lstm_units)(attr_features)

attr_features = tf.reshape(attr_features, [P.batch_size, P.attr_num, P.lstm_units])

attr_features = self_attention_layer(P.attention_size)(attr_features)

features = layers.Concatenate([title_features, attr_features])
features = layers.Dense(32, activation="relu")(features)
out_put = layers.Dense(P.num_class, activation='softmax')
model = keras.Model(inputs=[title_input, attr_input], outputs=out_put, name='name and attr model')

model.summary()

# instance_input = keras.Input(shape=(None,), name='instance')
# instance_features = embedding_layer(instance_input)
# instance_attention_layer = self_attention_layer(P.char_embed_size)
# instance_features = instance_attention_layer(instance_features)
#
# try:
#     pass
# except Exception:
#     print(Exception.__traceback__)