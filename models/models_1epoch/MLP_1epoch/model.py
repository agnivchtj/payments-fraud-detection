# model.py

from sklearn.neural_network import MLPClassifier
from joblib import dump


def load():
    # Neural Networks multi-layer perceptron
    clf_mlp = MLPClassifier()
    dump(clf_mlp, 'mlp_default.joblib')


if __name__ == '__main__':
    load()
