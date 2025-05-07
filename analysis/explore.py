from run_chil_exp import read_data

import numpy as np


def explore(dataset, random_seed, mimic_size):
    X_train, X_test, y_train, y_test = read_data(dataset, random_seed, mimic_size)
    print(dataset)
    print(X_train.shape)
    print(y_train.shape, y_train.min(), y_train.max())

    print(X_test.shape)
    print(y_test.shape)

    print(np.bincount(y_train) + np.bincount(y_test))
    print("*" * 50)


mimic_size = "full_mimic3"
explore("mimic3_mort_hosp", 42, mimic_size)
explore("mimic3_mort_icu", 42, mimic_size)
explore("mimic3_los_3", 42, mimic_size)
explore("mimic3_los_7", 42, mimic_size)
