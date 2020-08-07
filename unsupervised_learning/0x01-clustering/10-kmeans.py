#!/usr/bin/env python3
"""
This module has the kmeans(X, k)
"""
import sklearn.cluster


def kmeans(X, k):
    """
    This method performs K-means on a dataset
    X is a numpy.ndarray of shape (n, d) containing the dataset
    k is the number of clusters
    Returns: C, clss
    C is a numpy.ndarray of shape (k, d) containing the centroid
    means for each cluster
    clss is a numpy.ndarray of shape (n,) containing the index of
    the cluster in C that each data point belongs to
    """
    kmean = sklearn.cluster.KMeans(n_clusters=k)
    kmean.fit(X)
    C = kmean.cluster_centers_
    clss = kmean.labels_
    return C, clss
