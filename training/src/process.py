import hydra
import pandas as pd
from hydra.utils import to_absolute_path as abspath
from omegaconf import DictConfig
from patsy import dmatrices
from sklearn.model_selection import train_test_split
from math import radians, cos, sin, asin, sqrt
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib


def get_data(raw_path: str):
    data = pd.read_csv(raw_path)
    return data


def get_features(target: str, features: list, data: pd.DataFrame):
    feature_str = " + ".join(features)
    y, X = dmatrices(
        f"{target} ~ {feature_str} - 1", data=data, return_type="dataframe"
    )
    return y, X


def rename_columns(X: pd.DataFrame):
    X.columns = X.columns.str.replace(" ", "", regex=True)
    return X

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
@hydra.main(version_base=None, config_path="../../config", config_name="main")
def process_data(config: DictConfig):
    """Function to process the data"""

    data = get_data(abspath(config.raw.path))

    data['Weatherconditions'] = data['Weatherconditions'].map(lambda x: str(x)[11:])
    data['Time_taken(min)'] = data['Time_taken(min)'].map(lambda x: str(x)[6:])
    for i in data.columns:
        data[i].loc[data[i] == 'NaN '] = np.nan
        data[i].loc[data[i] == 'NaN'] = np.nan
    # delete missing values in Time_Orderd column
    data.dropna(subset=['Time_Orderd'], axis=0, inplace=True)

    # fill the missing values with their forward values
    data = data.fillna(method='ffill')
    features = ['Delivery_person_Age', 'Delivery_person_Ratings', 'multiple_deliveries', 'Time_taken(min)']
    for i in features:
        data[i] = data[i].astype(str).astype(float)
    data['Distance'] = data.apply(lambda x: haversine(x['Restaurant_latitude'], x['Restaurant_longitude'],
                                    x['Delivery_location_latitude'], x['Delivery_location_longitude']), axis=1)


    data = data[['Delivery_person_Age', 'Delivery_person_Ratings', 'Distance', 'Type_of_order',
                    'Type_of_vehicle', 'Time_taken(min)']]
    int_cols = data.select_dtypes('int')
    float_cols = data.select_dtypes('float')
    str_cols = data.select_dtypes(object)
    target_col = ['Time_taken(min)']
    data = pd.get_dummies(data, columns=str_cols.columns.tolist())
    scaler = StandardScaler()
    scaler = scaler.fit(data[target_col].values.reshape((-1,1)))
    data[target_col] = scaler.transform(data[target_col].values.reshape((-1,1)))
    data.rename(columns = {'Time_taken(min)':'Time_taken'}, inplace = True)
    data = rename_columns(data)
    y, X = get_features(config.process.target, config.process.features, data)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )


    # Save data
    X_train.to_csv(abspath(config.processed.X_train.path), index=False)
    X_test.to_csv(abspath(config.processed.X_test.path), index=False)
    y_train.to_csv(abspath(config.processed.y_train.path), index=False)
    y_test.to_csv(abspath(config.processed.y_test.path), index=False)

    #Save scaler
    joblib.dump(scaler, abspath(config.artefacts.path)) 

if __name__ == "__main__":
    process_data()