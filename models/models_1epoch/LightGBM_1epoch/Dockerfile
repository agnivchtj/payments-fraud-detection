FROM jupyter/scipy-notebook

RUN pip install joblib
RUN pip install lightgbm

COPY model.py ./model.py

RUN python3 model.py