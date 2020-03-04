from data import get_train_data
import tensorflow as tf
import  numpy as np

def lstm_learner(batch_embeddings):
    # batchsize * embedding size * time step
    X_origin = tf.reshape(batch_embeddings, [-1, 200])
    x_in = tf.matmul(X_origin, w1) + b1
    x_in_2 = tf.reshape(x_in, [-1, 25, 128])
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(128)
    init_state = lstm_cell.zero_state(batch_size=100, dtype=tf.float32)
    outputs, states = tf.nn.dynamic_rnn(lstm_cell, x_in_2, initial_state= init_state, time_major=False)
    results = tf.matmul(states[1], w2) + b2
    return results


def next_batch(x, y, step, batchsize):
    st = batchsize * step
    ed = batchsize * (step + 1) if batchsize * (step + 1) < len(x) else len(x)
    return x[st:ed], y[st:ed]


if __name__ == '__main__':
    train_table, train_label, test_table, test_label = get_train_data()

    x = tf.placeholder(tf.float32, [None, 200, 25])
    y = tf.placeholder(tf.float32, [None, 20])

    w1 = tf.Variable(tf.random_normal([25, 128]))
    w2 = tf.Variable(tf.random_normal([128, 20]))
    b1 = tf.Variable(tf.constant(0.1, shape=[128]))
    b2 = tf.Variable(tf.constant(0.1, shape=[20]))

    pred = lstm_learner(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    train_op = tf.train.AdamOptimizer(0.001).minimize(cost)
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    tf.initialize_all_variables()
    sess = tf.Session()

    for i in range(1000):
        step_num = int(len(train_label) / 100)
        for s in range(step_num):
            batch_xs, batch_ys = next_batch(train_table, train_label, s, 100)
            batch_xs = batch_xs.reshape([100, 25, 200])
            sess.run([train_op], feed_dict={
                x: batch_xs,
                y: batch_ys,
            })

        if i % 20 == 0:
            print(sess.run(accuracy, feed_dict={
                x: test_table,
                y: test_table,
            }))

    np.eye(20, dtype=np.float32)[y[st:ed]]