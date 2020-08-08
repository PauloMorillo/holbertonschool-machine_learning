#!/usr/bin/env python3

import numpy as np
EM = __import__('8-EM').expectation_maximization

if __name__ == "__main__":
    X = np.random.rand(100, 3)
    print(EM(X, 3, '5'))
    print(EM(X, 4, 0))
    print(EM(X, 5, -4))