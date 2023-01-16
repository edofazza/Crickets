from tensorflow import keras as ks
import numpy as np
import tensorflow as tf
import atexit
import gc
from argparse import ArgumentParser


class GeneticSearch:
    LOSS = 'binary_crossentropy'

    def __init__(self, train_set, train_labels, val_set, val_labels, batch_size=16, epochs=1000, shape=(10, 870)):
        self.train_set = train_set
        self.train_labels = train_labels
        self.val_set = val_set
        self.val_labels = val_labels
        self.batch_size = batch_size
        self.epochs = epochs
        self.shape = shape
        self.already_trained = dict()

    def _build_model(self, gene):
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

        # Conv layers
        for i in range(0, 25, 6):
            if int(gene[i]) == 0:
                continue
            x = ks.layers.Conv1D(
                int(gene[i+1]),
                3,
                padding='SAME'
            )(x)
            name += f'Conv{int(gene[i+1])}'
            if int(gene[i+2]) == 1:
                x = ks.layers.BatchNormalization()(x)
                name += 'Batch'
            x = ks.layers.Activation(activations[int(gene[i+3])])(x)
            name += activations[int(gene[i+3])]

            if int(gene[i+4]) == 1:
                dropout_rate = self._round_to_05(gene[i+5])
                if dropout_rate == 0.0:
                    continue
                x = ks.layers.Dropout(dropout_rate)(x)
                name += f'Drop{dropout_rate}'

        # Middle layer
        if int(gene[30]) == 0:
            x = ks.layers.Flatten()(x)
            name += 'Flatten'
        else:
            x = ks.layers.GlobalAveragePooling1D()(x)
            name += 'Global'

        # FCC layers
        for i in range(31, 52, 5):
            if int(gene[i]) == 0:
                continue
            x = ks.layers.Dense(
                int(gene[i + 1]),
                activation=activations[int(gene[i + 2])]
            )(x)
            name += f'Dense{int(gene[i + 1])}{activations[int(gene[i + 2])]}'

            if int(gene[i + 3]) == 1:
                dropout_rate = self._round_to_05(gene[i+4])
                if dropout_rate == 0.0:
                    continue
                x = ks.layers.Dropout(dropout_rate)(x)
                name += f'Drop{dropout_rate}'

        outputs = ks.layers.Dense(1, activation='sigmoid')(x)
        model = ks.Model(inputs, outputs)
        model.compile(
            optimizer='adam',
            loss=self.LOSS,
            metrics=['accuracy']
        )
        return name, model

    def _build_train(self, gene):
        # check feasibility to avoid to start constructing
        if int(gene[0]) == 0 and int(gene[6]) == 0 and int(gene[12]) == 0\
                and int(gene[18]) == 0 and int(gene[24]) == 0:
            return 20


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
                save_best_only=True,
                restore_best_weights=True
            ),
            ks.callbacks.ReduceLROnPlateau(
                patience=50,
                factor=0.5,
                min_delta=1e-3,
            )
        ]
        print(name)
        model.summary()

        #options = tf.data.Options()
        #options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
        #train_data = self.train_set.with_options(options)
        #val_data = self.val_set.with_options(options)

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
        gc.collect()
        model = ks.models.load_model(name + '.keras')
        train_loss, train_accuracy = model.evaluate(self.train_set, self.train_labels)
        val_loss, val_accuracy = model.evaluate(self.val_set, self.val_labels)
        print(
            f'{name}:\n\tTrain loss: {train_loss}\n\tTrain accuracy: {train_accuracy}\n\tVal loss: {val_loss}\n\tVal accuracy: {val_accuracy}')
        del model
        ks.backend.clear_session()

        if train_accuracy < 0.6 or val_accuracy < 0.6 or train_accuracy < val_accuracy:
            return 10

        self.already_trained[name] = result
        return result

    def _round_to_05(self, x):
        return round(x*20) / 20

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
                         np.load(opt['val_lab_path']),)
    test.get_val_loss(weights)

