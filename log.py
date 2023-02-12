import mlflow
import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt 

import mlflow
import mlflow.xgboost
from mlflow.models.signature import infer_signature
from hyperopt import (
    fmin, 
    hp, 
    tpe, 
    rand, 
    SparkTrials, 
    Trials, 
    STATUS_OK
)
from sklearn.metrics import (
    mean_squared_error,
    r2_score
)
from hyperopt.pyll.base import scope
import os
os.environ['MLFLOW_TRACKING_USERNAME'] = 'mohamedzayyan'
os.environ['MLFLOW_TRACKING_PASSWORD'] = '993b6e6575dafc00dc0781e647b9e8378a87c1be'

mlflow.set_tracking_uri('https://dagshub.com/mohamedzayyan/Delivery-time-prediction.mlflow')
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
    return c * r
with mlflow.start_run():
    # code to train model 
    trainData = pd.read_csv('./data/train.csv')
    testData = pd.read_csv('./data/test.csv')

    trainData['Weatherconditions'] = trainData['Weatherconditions'].map(lambda x: str(x)[11:])
    testData['Weatherconditions'] = testData['Weatherconditions'].map(lambda x: str(x)[11:])

    trainData['Time_taken(min)'] = trainData['Time_taken(min)'].map(lambda x: str(x)[6:])
    for i in trainData.columns:
        trainData[i].loc[trainData[i] == 'NaN '] = np.nan
        trainData[i].loc[trainData[i] == 'NaN'] = np.nan

    for j in testData.columns:
        testData[j].loc[testData[j] == 'NaN '] = np.nan
        testData[j].loc[testData[j] == 'NaN'] = np.nan
  
    # delete missing values in Time_Orderd column
    trainData.dropna(subset=['Time_Orderd'], axis=0, inplace=True)
    testData.dropna(subset=['Time_Orderd'], axis=0, inplace=True)

    # fill the missing values with their forward values
    trainData = trainData.fillna(method='ffill')
    testData = testData.fillna(method='ffill')
    features = ['Delivery_person_Age', 'Delivery_person_Ratings', 'multiple_deliveries', 'Time_taken(min)']
    features1 =  ['Delivery_person_Age', 'Delivery_person_Ratings', 'multiple_deliveries']
    for i in features:
        trainData[i] = trainData[i].astype(str).astype(float)
        for j in features1:
            testData[j] = testData[j].astype(str).astype(float)
    trainData['Distance(km)'] = trainData.apply(lambda x: haversine(x['Restaurant_latitude'], x['Restaurant_longitude'],
                                    x['Delivery_location_latitude'], x['Delivery_location_longitude']), axis=1)

    testData['Distance(km)'] = testData.apply(lambda x: haversine(x['Restaurant_latitude'], x['Restaurant_longitude'],
                                    x['Delivery_location_latitude'], x['Delivery_location_longitude']), axis=1)
    new_train_data = trainData[['Delivery_person_Age', 'Delivery_person_Ratings', 'Distance(km)', 'Type_of_order',
                    'Type_of_vehicle', 'Time_taken(min)']]
    int_cols = new_train_data.select_dtypes('int')
    float_cols = new_train_data.select_dtypes('float')
    str_cols = new_train_data.select_dtypes(object)
    target_col = ['Time_taken(min)']
    new_train_data = pd.get_dummies(new_train_data, columns=str_cols.columns.tolist())
    scaler = StandardScaler()
    scaler = scaler.fit(new_train_data[target_col].values.reshape((-1,1)))
    new_train_data[target_col] = scaler.transform(new_train_data[target_col].values.reshape((-1,1)))
    y = new_train_data.pop(target_col[0]).values
    X = new_train_data.values

    X_train, X_val, y_train, y_val = train_test_split(X,y, test_size=0.2, random_state=42)
    # Setting search space for xgboost model
    search_space = {
    'objective': 'reg:squarederror',
    'tree_method': 'gpu_hist',
    'eval_metric': 'r2',
    'max_depth': scope.int(hp.quniform('max_depth', 2, 10, 1)),
    'n_estimators': scope.int(hp.quniform('n_estimators', 60, 200, 40)),
    'learning_rate': hp.loguniform('max_depth', -7, 0),
    'min_child_weight': hp.loguniform('min_child_weight', -1, 7),
    'reg_alpha': hp.loguniform('reg_alpha', -10, 10),
    'reg_lambda': hp.loguniform('reg_lambda', -10, 10),
    'gamma': hp.loguniform('gamma', -10, 10),
    'use_label_encoder': False,
    'verbosity': 0,
    'random_state': 42
    }

    try:
        EXPERIMENT_ID = mlflow.create_experiment('xgboost-hyperopt')
    except:
        EXPERIMENT_ID = dict(mlflow.get_experiment_by_name('xgboost-hyperopt'))['experiment_id']
        
    def train_model(params):
        """
        Creates a hyperopt training model funciton that sweeps through params in a nested run
        Args:
            params: hyperparameters selected from the search space
        Returns:
            hyperopt status and the loss metric value
        """
        # With MLflow autologging, hyperparameters and the trained model are automatically logged to MLflow.
        # This sometimes doesn't log everything you may want so I usually log my own metrics and params just in case
        mlflow.xgboost.autolog()

        # 
        with mlflow.start_run(experiment_id=EXPERIMENT_ID, nested=True):
            # Training xgboost classifier
            model = XGBRegressor(**params)
            model = model.fit(X_train, y_train)

            # Predicting values for training and validation data, and getting prediction probabilities
            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)

            # Evaluating model metrics for training set predictions and validation set predictions
            # Creating training and validation metrics dictionaries to make logging in mlflow easier
            metric_names = ['mse', 'r2']
            # Training evaluation metrics
            train_mse = mean_squared_error(y_train, y_train_pred).round(3)
            train_r2 = r2_score(y_train, y_train_pred).round(3)
            training_metrics = {
                'mse': train_mse, 
                'r2': train_r2, 
            }
            training_metrics_values = list(training_metrics.values())

            # Validation evaluation metrics
            val_mse = mean_squared_error(y_val, y_val_pred).round(3)
            val_r2 = r2_score(y_val, y_val_pred).round(3)
            validation_metrics = {
                'mse': val_mse, 
                'r2': val_r2, 
            }
            validation_metrics_values = list(validation_metrics.values())

            # Logging model signature, class, and name
            signature = infer_signature(X_train, y_val_pred)
            mlflow.xgboost.log_model(model, 'model', signature=signature)
            mlflow.set_tag('estimator_name', model.__class__.__name__)
            mlflow.set_tag('estimator_class', model.__class__)

            # Logging each metric
            for name, metric in list(zip(metric_names, training_metrics_values)):
                mlflow.log_metric(f'training_{name}', metric)
            for name, metric in list(zip(metric_names, validation_metrics_values)):
                mlflow.log_metric(f'validation_{name}', metric)

            # Set the loss to -1*validation auc roc so fmin maximizes the it
            return {'status': STATUS_OK, 'loss': -1*validation_metrics['r2']}
    # Greater parallelism will lead to speedups, but a less optimal hyperparameter sweep.
    # A reasonable value for parallelism is the square root of max_evals.
    # spark_trials = SparkTrials()
    # Will need spark configured and installed to run. Add this to fmin function below like so:
    # trials = spark_trials
    trials = Trials()

    # Run fmin within an MLflow run context so that each hyperparameter configuration is logged as a child run of a parent
    # run called "xgboost_models" .
    with mlflow.start_run(experiment_id=EXPERIMENT_ID, run_name='xgboost_models', nested=True):
        xgboost_best_params = fmin(
            fn=train_model, 
            space=search_space, 
            algo=tpe.suggest,
            trials=trials,
            max_evals=50
        )