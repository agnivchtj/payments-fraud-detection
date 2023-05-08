# model.py

from sklearn.ensemble import RandomForestClassifier
from joblib import dump


def load():
    # Random Forest classifier
    clf_rf = RandomForestClassifier()
    dump(clf_rf, 'rf_default.joblib')


if __name__ == '__main__':
    load()
