from digit_recognizer import TRAIN

import numpy as np
import pandas as pd


def main():
    labels = np.loadtxt(TRAIN, delimiter=',', skiprows=1, usecols=0, dtype=np.int32)
    labels = pd.Series(labels)
    print(labels.value_counts())


if __name__ == '__main__':
    main()
