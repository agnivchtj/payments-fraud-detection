FROM jupyter/scipy-notebook

RUN pip install joblib
RUN pip install catboost

COPY model.py ./model.py

RUN python3 model.py