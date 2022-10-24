PRESENT_LOWER_BOUND = 0.0
PRESENT_UPPER_BOUND = 1.99

FILTERS_LOWER_BOUND = 8.0
FILTERS_UPPER_BOUND = 87.99

DROPRATE_LOWER_BOUND = 0.0
DROPRATE_UPPER_BOUND = 0.5

MIDDLELAYER_LOWER_BOUND = 0.0
MIDDLELAYER_UPPER_BOUND = 1.99

UNITS_LOWER_BOUND = 3.00
UNITS_UPPER_BOUND = 128.99

LOWER_BOUNDS = [PRESENT_LOWER_BOUND, # Conv1D 1 0
                FILTERS_LOWER_BOUND,
                PRESENT_LOWER_BOUND,
                DROPRATE_LOWER_BOUND,
                PRESENT_LOWER_BOUND, # Conv1D 2 4
                FILTERS_LOWER_BOUND,
                PRESENT_LOWER_BOUND,
                DROPRATE_LOWER_BOUND,
                PRESENT_LOWER_BOUND, # Conv1D 3 8
                FILTERS_LOWER_BOUND,
                PRESENT_LOWER_BOUND,
                DROPRATE_LOWER_BOUND,
                MIDDLELAYER_LOWER_BOUND, # Middle Layer 12
                PRESENT_LOWER_BOUND, # FC 1 13
                UNITS_LOWER_BOUND,
                PRESENT_LOWER_BOUND,
                DROPRATE_LOWER_BOUND,
                PRESENT_LOWER_BOUND, # FC 2 17
                UNITS_LOWER_BOUND,
                PRESENT_LOWER_BOUND,
                DROPRATE_LOWER_BOUND,
                PRESENT_LOWER_BOUND, # FC 3 21
                UNITS_LOWER_BOUND,
                PRESENT_LOWER_BOUND,
                DROPRATE_LOWER_BOUND]

UPPER_BOUNDS = [PRESENT_UPPER_BOUND, # Conv1D 1
                FILTERS_UPPER_BOUND,
                PRESENT_UPPER_BOUND,
                DROPRATE_UPPER_BOUND,
                PRESENT_UPPER_BOUND, # Conv1D 2
                FILTERS_UPPER_BOUND,
                PRESENT_UPPER_BOUND,
                DROPRATE_UPPER_BOUND,
                PRESENT_UPPER_BOUND, # Conv1D 3
                FILTERS_UPPER_BOUND,
                PRESENT_UPPER_BOUND,
                DROPRATE_UPPER_BOUND,
                MIDDLELAYER_UPPER_BOUND, # Middle Layer
                PRESENT_UPPER_BOUND, # FC 1
                UNITS_UPPER_BOUND,
                PRESENT_UPPER_BOUND,
                DROPRATE_UPPER_BOUND,
                PRESENT_UPPER_BOUND, # FC 2
                UNITS_UPPER_BOUND,
                PRESENT_UPPER_BOUND,
                DROPRATE_UPPER_BOUND,
                PRESENT_UPPER_BOUND, # FC 3
                UNITS_UPPER_BOUND,
                PRESENT_UPPER_BOUND,
                DROPRATE_UPPER_BOUND]


if __name__=='__main__':
    print(len(UPPER_BOUNDS), len(LOWER_BOUNDS))