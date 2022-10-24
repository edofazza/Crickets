from tensorflow import keras as ks
import numpy as np
import tensorflow as tf
import atexit
from argparse import ArgumentParser

class GeneticSearch:
    LOSS = 'sparse_categorical_crossentropy'

    def __init__(self, train_set, train_l, val_set, val_l, epochs=200, shape=(10, 87)):
        self.train_set = train_set
        self.train_labels = train_l
        self.val_set = val_set
        self.val_labels = val_l
        self.epochs = epochs
        self.shape = shape
        self.already_trained = dict()

    def _build_model(self, gene):
        name = ''
        data_augmentation = ks.Sequential(
            [
                ks.layers.GaussianNoise(0.5)
            ]
        )

        inputs = ks.Input(shape=self.shape)

        x = data_augmentation(inputs)
        for i in range(0, 12, 4):  # 1D
            if int(gene[i]) == 0:  # not present
                continue
            x = ks.layers.Conv1D(int(gene[i + 1]), 3, activation='relu', padding='SAME')(x)
            name += f'Conv{int(gene[i + 1])}'
            if int(gene[i + 2]) == 1:
                x = ks.layers.Dropout(gene[i + 3])(x)
                name += f'Drop{gene[i + 3]}'

        # Middle Layer
        if int(gene[12]) == 0:  # Flatten
            x = ks.layers.Flatten()(x)
            name += 'Flatten'
        else:
            x = ks.layers.GlobalAveragePooling1D()(x)
            name += 'Global'

        for i in range(13, 22, 4):  # FC
            if int(gene[i]) == 0:  # not present
                continue
            x = ks.layers.Dense(int(gene[i + 1]),
                                activation='relu')(x)
            name += f'Dense{int(gene[i + 1])}'
            if int(gene[i + 2]) == 1:
                x = ks.layers.Dropout(gene[i + 3])(x)
                name += f'Drop{gene[i + 3]}'

        outputs = ks.layers.Dense(3, activation='softmax')(x)
        model = ks.Model(inputs, outputs)
        model.compile(
            optimizer='adam',
            loss=self.LOSS,
            metrics=['accuracy']
        )
        return name, model

    def _build_train(self, gene):
        mirrored_strategy = tf.distribute.MirroredStrategy()

        # check feasibility to avoid to start constructing
        if gene[0] == 0 and gene[4] == 0 and gene[8] == 0:
            return 20

        with mirrored_strategy.scope():
            name, model = self._build_model(gene)

        if name in self.already_trained:
            return self.already_trained[name]

        callback_list = [
            ks.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
            ),
            ks.callbacks.ModelCheckpoint(
                filepath=f'{name}.keras',
                monitor='val_loss',
                save_best_only=True,
                restore_best_weights=True
            ),
            ks.callbacks.ReduceLROnPlateau(
                patience=5,
                factor=0.5,
                min_delta=1e-3,
            )
        ]
        model.summary()
        print(name)
        train_data = tf.data.Dataset.from_tensor_slices((self.train_set, self.train_labels))
        val_data = tf.data.Dataset.from_tensor_slices((self.val_set, self.val_labels))

        batch_size = 32
        train_data = train_data.batch(batch_size)
        val_data = val_data.batch(batch_size)

        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
        train_data = train_data.with_options(options)
        val_data = val_data.with_options(options)

        history = model.fit(
            train_data,
            validation_data=val_data,
            epochs=self.epochs,
            callbacks=callback_list,
            verbose=0
        )
        del model
        ks.backend.clear_session()
        result = float(np.min(history.history['val_loss']))
        self.already_trained[name] = result
        atexit.register(mirrored_strategy._extended._collective_ops._pool.close)
        return result

    def get_val_accuracy(self, gene):
        return self._build_model(gene)


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
    test.get_val_accuracy(weights)

