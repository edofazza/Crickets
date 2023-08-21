import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Activation, Add, GlobalAveragePooling1D, Dense
import tensorflow as tf


# Basic 1D Residual Block
def basic_residual_block(x, filters, kernel_size, stride=1, activation='relu', batch_norm=True, conv_first=True):
    if conv_first:
        x = Conv1D(filters, kernel_size=kernel_size, strides=stride, padding='same')(x)
        if batch_norm:
            x = BatchNormalization()(x)
        if activation:
            x = Activation(activation)(x)

    else:
        if batch_norm:
            x = BatchNormalization()(x)
        if activation:
            x = Activation(activation)(x)
        x = Conv1D(filters, kernel_size=kernel_size, strides=stride, padding='same')(x)

    return x


# ResNet-18
def resnet18(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = inputs

    x = basic_residual_block(x, filters=64, kernel_size=7, stride=2)
    x = tf.keras.layers.MaxPooling1D(pool_size=3, strides=2, padding="same")(x)

    x = basic_residual_block(x, filters=64, kernel_size=3)
    x = basic_residual_block(x, filters=64, kernel_size=3)

    x = basic_residual_block(x, filters=128, kernel_size=3, stride=2)
    x = basic_residual_block(x, filters=128, kernel_size=3)

    x = basic_residual_block(x, filters=256, kernel_size=3, stride=2)
    x = basic_residual_block(x, filters=256, kernel_size=3)

    x = basic_residual_block(x, filters=512, kernel_size=3, stride=2)
    x = basic_residual_block(x, filters=512, kernel_size=3)

    x = GlobalAveragePooling1D()(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    return model


# ResNet-34
def resnet34(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = inputs

    x = basic_residual_block(x, filters=64, kernel_size=7, stride=2)
    x = tf.keras.layers.MaxPooling1D(pool_size=3, strides=2, padding="same")(x)

    for _ in range(3):
        x = basic_residual_block(x, filters=64, kernel_size=3)

    x = basic_residual_block(x, filters=128, kernel_size=3, stride=2)
    for _ in range(4):
        x = basic_residual_block(x, filters=128, kernel_size=3)

    x = basic_residual_block(x, filters=256, kernel_size=3, stride=2)
    for _ in range(6):
        x = basic_residual_block(x, filters=256, kernel_size=3)

    x = basic_residual_block(x, filters=512, kernel_size=3, stride=2)
    for _ in range(3):
        x = basic_residual_block(x, filters=512, kernel_size=3)

    x = GlobalAveragePooling1D()(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    return model


# ResNet-50
def resnet50(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = inputs

    x = basic_residual_block(x, filters=64, kernel_size=7, stride=2)
    x = tf.keras.layers.MaxPooling1D(pool_size=3, strides=2, padding="same")(x)

    x = basic_residual_block(x, filters=64, kernel_size=1)
    x = basic_residual_block(x, filters=64, kernel_size=3)
    x = basic_residual_block(x, filters=256, kernel_size=1)

    x = basic_residual_block(x, filters=128, kernel_size=1)
    x = basic_residual_block(x, filters=128, kernel_size=3)
    x = basic_residual_block(x, filters=512, kernel_size=1)

    x = basic_residual_block(x, filters=256, kernel_size=1)
    x = basic_residual_block(x, filters=256, kernel_size=3)
    x = basic_residual_block(x, filters=1024, kernel_size=1)

    x = basic_residual_block(x, filters=512, kernel_size=1)
    x = basic_residual_block(x, filters=512, kernel_size=3)
    x = basic_residual_block(x, filters=2048, kernel_size=1)

    x = GlobalAveragePooling1D()(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    return model


if __name__=='__main__':
    # Model parameters
    input_shape = (256, 8)  # Adjust input shape according to your data
    num_classes = 3  # Number of classes in your classification task

    # Create the models
    resnet18_model = resnet18(input_shape, num_classes)