from celery import Celery
import numpy as np
import joblib

# Celery configuration
CELERY_BROKER_URL = 'amqp://rabbitmq:rabbitmq@rabbitmq:5672/'
CELERY_RESULT_BACKEND = 'rpc://'

# Initialize Celery
celery = Celery('workerA', broker=CELERY_BROKER_URL, backend=CELERY_RESULT_BACKEND)

# Load the model once (on worker startup)
model = joblib.load('final_model.pkl')

@celery.task
def get_predictions(X_input):
    """
    Accepts a 2D list of feature values (X_input) and returns model predictions.
    Example: X_input = [[5, 116, 74, ..., 30], [...], ...]
    """
    X = np.array(X_input)
    predictions = model.predict(X)
    return predictions.tolist()
