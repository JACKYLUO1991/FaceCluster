from sklearn.cluster import DBSCAN
import multiprocessing as mp
import dlib


def dbscan(encodings, eps, min_samples, **kwargs):
    """
    """
    db = DBSCAN(eps=eps, min_samples=min_samples,
                n_jobs=mp.cpu_count()).fit(encodings)
    return db.labels_


def chinese_whispers(encodings, threshold=0.5):
    """
    Chinese Whispers - an Efficient Graph Clustering Algorithm 
    and its Application to Natural Language Processing Problems
    """
    encodings = [dlib.vector(enc) for enc in encodings]
    return dlib.chinese_whispers_clustering(encodings, threshold)

