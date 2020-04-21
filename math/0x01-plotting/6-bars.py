#!/usr/bin/env python3
""" This script plots a bar with offset """


import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruits = np.random.randint(0, 20, (4, 3))
index = ['Farrah', 'Fred', 'Felicia']
bar_width = 0.5
cell_text = []
colors = ['r', 'yellow', '#ff8000', '#ffe5b4']
labels = ['apples', 'bananas', 'oranges', 'peaches']
y_offset = np.zeros(3)
for row in range(len(fruits)):
    plt.bar(index, fruits[row], bar_width, bottom=y_offset,
            color=colors[row], label=labels[row])
    y_offset = y_offset + fruits[row]
    plt.ylabel('Quantity of Fruit')
    plt.title('Number of Fruit per Person')
    cell_text.reverse()
    plt.ylim(0, 80)
    plt.legend()
    plt.show()
