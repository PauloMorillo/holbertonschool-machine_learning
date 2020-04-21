#!/usr/bin/env python3
""" This script plots a scattering with a third variable like elevation """


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

np.random.seed(5)

x = np.random.randn(2000) * 10
y = np.random.randn(2000) * 10
z = np.random.rand(2000) + 40 - np.sqrt(np.square(x) + np.square(y))
plt.xlabel('x coordinate (m)')
plt.ylabel('y coordinate (m)')
plt.title('Mountain Elevation')
plt.scatter(x, y, c=z)
cob = plt.colorbar(orientation='vertical')
cob.set_label('elevation (m)')
plt.show()
