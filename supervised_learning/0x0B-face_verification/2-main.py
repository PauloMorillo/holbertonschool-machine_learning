#!/usr/bin/env python3

from align import FaceAlign

fa = FaceAlign('/models/shape_predictor_68_face_landmarks.dat')
print(type(fa.detector))
print(type(fa.shape_predictor))