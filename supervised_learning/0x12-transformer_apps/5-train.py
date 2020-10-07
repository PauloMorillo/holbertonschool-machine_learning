#!/usr/bin/env python3
"""
This module has the
def train_transformer(N, dm, h, hidden, max_len, batch_size, epochs): method
"""

Dataset = __import__('3-dataset').Dataset
create_masks = __import__('4-create_masks').create_masks
Transformer = __import__('5-transformer').Transformer


def train_transformer(N, dm, h, hidden, max_len, batch_size, epochs):
    """
    This method  creates and trains a transformer model for machine translation
    of Portuguese to English using our previously created dataset
    """
    pass
