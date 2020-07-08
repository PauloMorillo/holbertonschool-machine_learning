#!/usr/bin/env python3
""" This script has the methods load_image
    and load_csv
"""
import cv2
import numpy as np
import glob
import pandas as pd
import csv


def load_images(images_path, as_array=True):
    """This method load images"""
    if as_array is not True:
        images = []
        names = []
        for file in glob.glob(images_path + "/*.jpg"):
            img = cv2.imread(file)
            #print(img)
            if img is not None:
                images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                names.append(file[5:])
    else:
        images = [cv2.cvtColor(cv2.imread(file), cv2.COLOR_BAYER_BG2RGB) for file in glob.glob(images_path + '/*.jpg')]
        names = [file[5:] for file in glob.glob(images_path + '/*.jpg')]
    #print(names)
    return images, names

def load_csv(csv_path, params={}):
    """This method load a csv"""
    #data = pd.read_csv(csv_path)
    with open(csv_path, 'r', newline='', encoding="UTF-8") as file:
        reader = csv.reader(file, params)
        labels = list()
        for row in reader:
            labels.append(row)
    return labels

