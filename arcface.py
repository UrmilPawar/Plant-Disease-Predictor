import tensorflow as tf

#definng the loss function
def arcface_loss(y_true, y_pred, margin=0.5, scale=64):
    # Extracting the cosine similarity values from the predictions
    cos_t = y_pred
    sin_t = tf.math.sqrt(1 - tf.math.square(cos_t))

    # Calculate the threshold and margin values
    threshold = tf.math.cos(tf.constant(math.pi - margin))
    cos_m = tf.math.cos(tf.constant(margin))
    sin_m = tf.math.sin(tf.constant(margin))
    safe_margin = sin_m * margin

    # Calculate the modified cosine values using the margin
    cos_t_margin = tf.where(cos_t > threshold,
                            cos_t * cos_m - sin_t * sin_m,
                            cos_t - safe_margin)

    # Apply one-hot encoding to the true labels
    mask = y_true
    cos_t_onehot = cos_t * mask
    cos_t_margin_onehot = cos_t_margin * mask

    # Calculate the final logits
    logits = (cos_t + cos_t_margin_onehot - cos_t_onehot) * scale

    # Compute softmax cross-entropy loss
    losses = tf.nn.softmax_cross_entropy_with_logits(y_true, logits)

    return losses