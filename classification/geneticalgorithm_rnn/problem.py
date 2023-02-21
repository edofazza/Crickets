from tensorflow import keras as ks
import numpy as np
import tensorflow as tf
import atexit
from argparse import ArgumentParser

class GeneticSearch:
    BINARY_LOSS = 'binary_crossentropy'
    LOSS = 'sparse_categorical_crossentropy'

    def __init__(self, train_set, train_labels, val_set, val_labels, batch_size=16, epochs=1000, shape=(8, 3480),
                 classes=2, multi_gpus=False):
        self.train_set = train_set
        self.train_labels = train_labels
        self.val_set = val_set
        self.val_labels = val_labels
        self.batch_size = batch_size
        self.epochs = epochs
        self.shape = shape
        self.classes = classes
        self.multi_gpus = multi_gpus
        self.already_trained = dict()

    def _build_model(self, gene):
        rnn_layer = ['lstm', 'gru']
        bi_layer = ['', 'bi']
        activations = [
            'sigmoid', 'swish', 'tanh', 'relu', 'gelu', 'elu', 'leaky_relu'
        ]

        name = ''
        data_augmentation = ks.Sequential(
            [
                ks.layers.GaussianNoise(0.5)
            ]
        )

        inputs = ks.Input(shape=self.shape)

        x = data_augmentation(inputs)
        """
        Check the present parameters and create array containing the index of
        all presents
        """
        # Check with RNN layer are present
        present_rnn_layers = [i for i in range(0, 21, 5) if int(gene[i]) == 1]
        for layer, i in enumerate(present_rnn_layers):
            if int(gene[i + 1]) == 1:  # Bidirectional
                if int(gene[i + 2]) == 0:  # LSTM
                    x = ks.layers.Bidirectional(
                        ks.layers.LSTM(
                            int(gene[i + 3]),
                            activation=activations[int(gene[i + 4])],
                            return_sequences=False if layer == len(present_rnn_layers) - 1 else True
                        ))(x)
                    """else:  # case with Combined Hyperbolic Sine
                        x = ks.layers.Bidirectional(
                            ks.layers.LSTM(
                                int(gene[i + 3]),
                                activation=combined_hyperbolic_sine,
                                return_sequences=False if layer == len(present_rnn_layers) - 1 else True
                            ))(x)"""
                else:  # GRU
                    #if int(gene[i + 4]) != 4:
                    x = ks.layers.Bidirectional(
                        ks.layers.GRU(
                            int(gene[i + 3]),
                            activation=activations[int(gene[i + 4])],
                            return_sequences=False if layer == len(present_rnn_layers) - 1 else True
                        ))(x)
                    """else:  # case with Combined Hyperbolic Sine
                        x = ks.layers.Bidirectional(
                            ks.layers.GRU(
                                int(gene[i + 3]),
                                activation=combined_hyperbolic_sine,
                                return_sequences=False if layer == len(present_rnn_layers) - 1 else True
                            ))(x)"""
            else:
                #if int(gene[i + 2]) == 0:  # LSTM
                if int(gene[i + 4]) != 4:
                    x = ks.layers.LSTM(
                        int(gene[i + 3]),
                        activation=activations[int(gene[i + 4])],
                        return_sequences=False if layer == len(present_rnn_layers) - 1 else True
                    )(x)
                    """else:  # case with Combined Hyperbolic Sine
                        x = ks.layers.LSTM(
                            int(gene[i + 3]),
                            activation=combined_hyperbolic_sine,
                            return_sequences=False if layer == len(present_rnn_layers) - 1 else True
                        )(x)"""
                else:  # GRU
                    #if int(gene[i + 4]) != 4:
                    x = ks.layers.GRU(
                        int(gene[i + 3]),
                        activation=activations[int(gene[i + 4])],
                        return_sequences=False if layer == len(present_rnn_layers) - 1 else True
                    )(x)
                    """else:  # case with Combined Hyperbolic Sine
                        x = ks.layers.GRU(
                            int(gene[i + 3]),
                            activation=combined_hyperbolic_sine,
                            return_sequences=False if layer == len(present_rnn_layers) - 1 else True
                        )(x)"""
            name += f'{bi_layer[int(gene[i + 1])]}{rnn_layer[int(gene[i + 2])]}' \
                    f'{int(gene[i + 3])}{activations[int(gene[i + 4])]}'

        # FCC layers
        for i in range(25, 46, 5):
            if int(gene[i]) == 0:
                continue

            x = ks.layers.Dense(
                int(gene[i + 1]),
                activation=activations[int(gene[i + 2])]
            )(x)

            name += f'Dense{int(gene[i + 1])}{activations[int(gene[i + 2])]}'

            if int(gene[i + 3]) == 1:
                dropout_rate = self._round_to_05(gene[i + 4])
                if dropout_rate == 0.0:
                    continue
                x = ks.layers.Dropout(dropout_rate)(x)
                name += f'Drop{dropout_rate}'

        if self.classes == 2:
            outputs = ks.layers.Dense(1, activation='sigmoid')(x)
        else:
            outputs = ks.layers.Dense(3, activation='softmax')(x)
        model = ks.Model(inputs, outputs)
        model.compile(
            optimizer='adam',
            loss=self.LOSS if self.classes != 2 else self.BINARY_LOSS,
            metrics=['accuracy']
        )
        return name, model

    def _build_train(self, gene):
        # check feasibility to avoid to start constructing
        if int(gene[0]) == 0 and int(gene[5]) == 0 and int(gene[10]) == 0 \
                and int(gene[15]) == 0 and int(gene[20]) == 0:
            return 20

        if self.multi_gpus:  # allow multiple gpus
            mirrored_strategy = tf.distribute.MirroredStrategy()
            with mirrored_strategy.scope():
                name, model = self._build_model(gene)
        else:
            name, model = self._build_model(gene)

        if name in self.already_trained:
            return self.already_trained[name]

        callback_list = [
            ks.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=100,
            ),
            ks.callbacks.ModelCheckpoint(
                filepath=f'{name}.keras',
                monitor='val_loss',
                save_best_only=True
            ),
            ks.callbacks.ReduceLROnPlateau(
                patience=50,
                factor=0.5,
                min_delta=1e-3,
            )
        ]
        print(name)
        model.summary()

        if self.multi_gpus:
            options = tf.data.Options()
            options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
            train_data = tf.data.Dataset.from_tensor_slices((self.train_set, self.train_labels))
            val_data = tf.data.Dataset.from_tensor_slices((self.val_set, self.val_labels))
            train_data = train_data.with_options(options)
            val_data = val_data.with_options(options)
            history = model.fit(
                train_data,
                validation_data=val_data,
                epochs=self.epochs,
                callbacks=callback_list,
                verbose=0,
                batch_size=self.batch_size
            )
        else:
            history = model.fit(
                self.train_set,
                self.train_labels,
                validation_data=(self.val_set, self.val_labels),
                epochs=self.epochs,
                callbacks=callback_list,
                verbose=0,
                batch_size=self.batch_size
            )
        result = float(np.min(history.history['val_loss']))
        if np.isnan(result) or str(result).lower() == 'nan':
            return 20
        del model
        model = ks.models.load_model(name + '.keras')
        train_loss, train_accuracy = model.evaluate(self.train_set, self.train_labels)
        val_loss, val_accuracy = model.evaluate(self.val_set, self.val_labels)
        print(
            f'{name}:\n\tTrain loss: {train_loss}\n\tTrain accuracy: {train_accuracy}\n\tVal loss: {val_loss}\n\tVal accuracy: {val_accuracy}')
        del model
        ks.backend.clear_session()

        if train_accuracy <= 0.34 or val_accuracy <= 0.34 or train_accuracy < val_accuracy:
            return 10 * (1 - train_accuracy)

        self.already_trained[name] = result

        if self.multi_gpus:
            atexit.register(mirrored_strategy._extended._collective_ops._pool.close)
        return result

    def _round_to_05(self, x):
        return round(x * 20) / 20

    def get_val_loss(self, gene):
        return self._build_train(gene)


def parse():
    parser = ArgumentParser()
    parser.add_argument('--gene_npy_path', type=str, default='hof.npy')
    parser.add_argument('--train_set_path', type=str, default='train.npy')
    return vars(parser.parse_args())


if __name__ == '__main__':
    opt = parse()
    weights = np.load(opt['gene_npy_path'])[0]
    test = GeneticSearch(np.load(opt['train_set_path']),
                         np.load(opt['train_lab_path']),
                         np.load(opt['val_set_path']),
                         np.load(opt['val_lab_path']), )
    test.get_val_loss(weights)
