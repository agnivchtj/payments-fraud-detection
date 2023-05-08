FROM jupyter/scipy-notebook

RUN pip install joblib

COPY model.py ./model.py

RUN python3 model.py