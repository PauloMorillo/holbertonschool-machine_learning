#!/usr/bin/env python3

import numpy as np
initialize = __import__('4-initialize').initialize

if __name__ == "__main__":
    print(initialize('hello', 5))
    print(initialize(np.array([1, 2, 3, 4, 5]), 5))
    print(initialize(np.array([[[1, 2, 3, 4, 5]]]), 5))