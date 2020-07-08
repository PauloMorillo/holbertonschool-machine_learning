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
        self.detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor(shape_predictor_path)
