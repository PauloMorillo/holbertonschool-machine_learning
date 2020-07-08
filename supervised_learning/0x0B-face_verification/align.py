#!/usr/bin/env python3
""" This script has the class """
import cv2
import numpy as np
import glob
import pandas as pd
import csv
import dlib


class FaceAlign():
    def __init__(self, shape_predictor_path):
        """ All begins here """
        print(shape_predictor_path)
        sp = dlib.shape_predictor(shape_predictor_path)
        return sp
