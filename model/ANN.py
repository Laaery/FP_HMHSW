#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time: 2025/9/10 20:01
# @Author: LL
# @File：ANN.py
import os
import tempfile

import joblib
from annoy import AnnoyIndex
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from collections import Counter
from joblib import Parallel, delayed


class AnnoyKNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neighbors=5, n_trees=10, search_k=-1, metric='euclidean', n_jobs=1):
        self.n_neighbors = n_neighbors
        self.n_trees = n_trees
        self.search_k = search_k
        self.metric = metric
        self.n_jobs = n_jobs

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.X_train_ = X.copy()
        self.y_train_ = y.copy()

        self.classes_ = np.unique(y)

        # 支持的距离度量映射
        valid_metrics = {
            'euclidean': 'euclidean',
            'manhattan': 'manhattan',
            'cosine': 'angular',
            'hamming': 'hamming'
        }
        if self.metric not in valid_metrics:
            raise ValueError(f"Metric must be in {list(valid_metrics.keys())}")

        self.annoy_metric_ = valid_metrics[self.metric]
        f = X.shape[1]
        self.index_ = AnnoyIndex(f, self.annoy_metric_)

        for i, x in enumerate(X):
            self.index_.add_item(i, x.tolist())

        self.index_.build(self.n_trees)

        return self

    def predict(self, X):
        X = np.asarray(X)
        predictions = Parallel(n_jobs=self.n_jobs)(
            delayed(self._predict_single)(x) for x in X
        )
        return np.array(predictions)

    def _predict_single(self, x):
        neighbors = self.index_.get_nns_by_vector(
            x, self.n_neighbors, search_k=self.search_k
        )
        neighbor_labels = [self.y_train_[i] for i in neighbors]
        most_common = Counter(neighbor_labels).most_common(1)
        return most_common[0][0]

    def predict_proba(self, X):
        X = np.asarray(X)
        classes_ = np.unique(self.y_train_)
        n_classes = len(classes_)
        proba = []

        for x in X:
            neighbors = self.index_.get_nns_by_vector(
                x, self.n_neighbors, search_k=self.search_k
            )
            neighbor_labels = [self.y_train_[i] for i in neighbors]
            label_counts = Counter(neighbor_labels)

            probs = np.array([label_counts.get(cls, 0) for cls in classes_], dtype=float)

            total = probs.sum()
            if total > 0:
                probs /= total
            else:
                probs = np.ones(n_classes) / n_classes

            proba.append(probs)

        return np.array(proba)

    def save(self, folder):
        # 保存索引
        self.index_.save(os.path.join(folder, 'index.ann'))
        # 保存其他参数
        joblib.dump({
            'X_train_': self.X_train_,
            'y_train_': self.y_train_,
            'n_neighbors': self.n_neighbors,
            'n_trees': self.n_trees,
            'metric': self.metric,
            'annoy_metric_': self.annoy_metric_
        }, os.path.join(folder, 'metadata.pkl'))

    @classmethod
    def load(cls, folder):
        """从文件夹加载模型"""
        metadata = joblib.load(os.path.join(folder, 'metadata.pkl'))
        model = cls(**{k: v for k, v in metadata.items() if k in ['n_neighbors', 'n_trees', 'metric']})

        # 恢复数据
        model.X_train_ = metadata['X_train_']
        model.y_train_ = metadata['y_train_']
        model.annoy_metric_ = metadata['annoy_metric_']

        # 加载索引
        f = model.X_train_.shape[1]
        model.index_ = AnnoyIndex(f, model.annoy_metric_)
        model.index_.load(os.path.join(folder, 'index.ann'))

        return model