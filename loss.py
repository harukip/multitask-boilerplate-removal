import tensorflow as tf


class Custom_Cross_Entropy(tf.keras.losses.Loss):
    def __init__(self, class_weights):
        super(Custom_Cross_Entropy, self).__init__()
        self.class_weights = class_weights

    def __call__(self, y_true, y_pred):
        y_true_flatten = tf.reshape(y_true, [-1, 2])
        y_pred_flatten = tf.reshape(y_pred, [-1, 2])
        weights = tf.reduce_sum(
            tf.cast(self.class_weights, tf.float32) * tf.cast(y_true_flatten, tf.float32), axis=1)
        unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(
            y_true_flatten, y_pred_flatten)
        weighted_losses = unweighted_losses * weights
        return tf.reduce_mean(weighted_losses)
