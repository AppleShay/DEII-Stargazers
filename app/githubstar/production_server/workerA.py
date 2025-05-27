from celery import Celery
import numpy as np
import joblib

# Celery configuration
CELERY_BROKER_URL = 'amqp://rabbitmq:rabbitmq@rabbit:5672/'
CELERY_RESULT_BACKEND = 'redis://redis:6379/0'

# Initialize Celery
celery = Celery('workerA', broker=CELERY_BROKER_URL, backend=CELERY_RESULT_BACKEND)

# Load the model once (on worker startup)
model = joblib.load('final_model.pkl')

@celery.task
def get_predictions(X_input):
    X = np.array(X_input)
    predictions=[]
    for i in range(5):
        predictions.append(model.predict([X[i]]))
    return predictions
