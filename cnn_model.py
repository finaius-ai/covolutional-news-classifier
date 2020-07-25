import tensorflow as tf

from data import weights


def cnn_model(vocab_size):
    model = tf.keras.models.Sequential(
        [
            # tf.keras.layers.InputLayer(),
            tf.keras.layers.Embedding(vocab_size, 32, input_shape=(128,)),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Conv1D(16, 3, activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(len(weights.keys()))
        ]

    )
    model.compile(optimizer=tf.keras.optimizers.SGD(0.0001),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

    return model