"""
Activation functions:
    - sigmoid
    - swish
    - tanh
    - relu
    - gelu
    - elu
    - leaky_relu
"""
ACTIVATION_LOWER_BOUND = 0.0
ACTIVATION_UPPER_BOUND = 6.99

"""
Type of RNN layer (LSTM, GRU)
"""
RNN_LAYER_LOWER_BOUND = 0.0
RNN_LAYER_UPPER_BOUND = 1.99

"""
Present can be 0 (not present) or 1 (present)
"""
PRESENT_LOWER_BOUND = 0.0
PRESENT_UPPER_BOUND = 1.99

"""
RNN UNITS can be from 16 to 1024
"""
RNN_UNITS_LOWER_BOUND = 16.0
RNN_UNITS_UPPER_BOUND = 1024.99

"""
Dropout can be from 0 to 0.5 (only two decimals but multiple of 0.05, handled in problem)
"""
DROPRATE_LOWER_BOUND = 0.0
DROPRATE_UPPER_BOUND = 0.5099


"""
Units (dense layers) can be from 3 to 512
"""
UNITS_LOWER_BOUND = 3.00
UNITS_UPPER_BOUND = 512.99


LOWER_BOUNDS = [
    PRESENT_LOWER_BOUND,    # RNN 0 - 0
    PRESENT_LOWER_BOUND,
    RNN_LAYER_LOWER_BOUND,
    RNN_UNITS_LOWER_BOUND,
    ACTIVATION_LOWER_BOUND,

    PRESENT_LOWER_BOUND,  # RNN 1 - 5
    PRESENT_LOWER_BOUND,
    RNN_LAYER_LOWER_BOUND,
    RNN_UNITS_LOWER_BOUND,
    ACTIVATION_LOWER_BOUND,

    PRESENT_LOWER_BOUND,  # RNN 2 - 10
    PRESENT_LOWER_BOUND,
    RNN_LAYER_LOWER_BOUND,
    RNN_UNITS_LOWER_BOUND,
    ACTIVATION_LOWER_BOUND,

    PRESENT_LOWER_BOUND,  # RNN 3 - 15
    PRESENT_LOWER_BOUND,
    RNN_LAYER_LOWER_BOUND,
    RNN_UNITS_LOWER_BOUND,
    ACTIVATION_LOWER_BOUND,

    PRESENT_LOWER_BOUND,  # RNN 4 - 20
    PRESENT_LOWER_BOUND,
    RNN_LAYER_LOWER_BOUND,
    RNN_UNITS_LOWER_BOUND,
    ACTIVATION_LOWER_BOUND,

    PRESENT_LOWER_BOUND,    # dense 0 - 25
    UNITS_LOWER_BOUND,
    ACTIVATION_LOWER_BOUND,
    PRESENT_LOWER_BOUND,
    DROPRATE_LOWER_BOUND,

    PRESENT_LOWER_BOUND,    # dense 1 - 30
    UNITS_LOWER_BOUND,
    ACTIVATION_LOWER_BOUND,
    PRESENT_LOWER_BOUND,
    DROPRATE_LOWER_BOUND,

    PRESENT_LOWER_BOUND,    # dense 2 - 45
    UNITS_LOWER_BOUND,
    ACTIVATION_LOWER_BOUND,
    PRESENT_LOWER_BOUND,
    DROPRATE_LOWER_BOUND,

    PRESENT_LOWER_BOUND,    # dense 3 - 40
    UNITS_LOWER_BOUND,
    ACTIVATION_LOWER_BOUND,
    PRESENT_LOWER_BOUND,
    DROPRATE_LOWER_BOUND,

    PRESENT_LOWER_BOUND,    # dense 4 - 45
    UNITS_LOWER_BOUND,
    ACTIVATION_LOWER_BOUND,
    PRESENT_LOWER_BOUND,
    DROPRATE_LOWER_BOUND
]

UPPER_BOUNDS = [
    PRESENT_UPPER_BOUND,    # RNN 0 - 0
    PRESENT_UPPER_BOUND,
    RNN_LAYER_UPPER_BOUND,
    RNN_UNITS_UPPER_BOUND,
    ACTIVATION_UPPER_BOUND,

    PRESENT_UPPER_BOUND,  # RNN 1 - 5
    PRESENT_UPPER_BOUND,
    RNN_LAYER_UPPER_BOUND,
    RNN_UNITS_UPPER_BOUND,
    ACTIVATION_UPPER_BOUND,

    PRESENT_UPPER_BOUND,  # RNN 2 - 10
    PRESENT_UPPER_BOUND,
    RNN_LAYER_UPPER_BOUND,
    RNN_UNITS_UPPER_BOUND,
    ACTIVATION_UPPER_BOUND,

    PRESENT_UPPER_BOUND,  # RNN 3 - 15
    PRESENT_UPPER_BOUND,
    RNN_LAYER_UPPER_BOUND,
    RNN_UNITS_UPPER_BOUND,
    ACTIVATION_UPPER_BOUND,

    PRESENT_UPPER_BOUND,  # RNN 4 - 20
    PRESENT_UPPER_BOUND,
    RNN_LAYER_UPPER_BOUND,
    RNN_UNITS_UPPER_BOUND,
    ACTIVATION_UPPER_BOUND,

    PRESENT_UPPER_BOUND,    # dense 0 - 25
    UNITS_UPPER_BOUND,
    ACTIVATION_UPPER_BOUND,
    PRESENT_UPPER_BOUND,
    DROPRATE_UPPER_BOUND,

    PRESENT_UPPER_BOUND,    # dense 1 - 30
    UNITS_UPPER_BOUND,
    ACTIVATION_UPPER_BOUND,
    PRESENT_UPPER_BOUND,
    DROPRATE_UPPER_BOUND,

    PRESENT_UPPER_BOUND,    # dense 2 - 35
    UNITS_UPPER_BOUND,
    ACTIVATION_UPPER_BOUND,
    PRESENT_UPPER_BOUND,
    DROPRATE_UPPER_BOUND,

    PRESENT_UPPER_BOUND,    # dense 3 - 40
    UNITS_UPPER_BOUND,
    ACTIVATION_UPPER_BOUND,
    PRESENT_UPPER_BOUND,
    DROPRATE_UPPER_BOUND,

    PRESENT_UPPER_BOUND,    # dense 4 - 45
    UNITS_UPPER_BOUND,
    ACTIVATION_UPPER_BOUND,
    PRESENT_UPPER_BOUND,
    DROPRATE_UPPER_BOUND
]

if __name__ == '__main__':
    print(len(UPPER_BOUNDS), len(LOWER_BOUNDS))
