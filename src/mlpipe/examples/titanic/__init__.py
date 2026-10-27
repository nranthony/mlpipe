"""Titanic (Kaggle `train.csv`): a small, fast, real-data example for the viewing
surfaces (work/0013). Like the Numerai example it is a harness (ADR-0009), not a
competition entry. `mlpipe predict` scores Kaggle's `test.csv` into a
submission-shaped `PassengerId, Survived` frame, exercising the generic serve path.

    mlpipe run --pipeline titanic-example
    mlpipe predict --pipeline titanic-example --bundle <hash>
"""
