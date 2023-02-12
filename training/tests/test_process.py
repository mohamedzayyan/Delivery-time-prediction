import pandas as pd
from pandera import Check, Column, DataFrameSchema
from pytest_steps import test_steps

from training.src.process import get_features, rename_columns


@test_steps("get_features_step", "rename_columns_step")
def test_processs_suite(test_step, steps_data):
    if test_step == "get_features_step":
        get_features_step(steps_data)
    elif test_step == "rename_columns_step":
        rename_columns_step(steps_data)


def get_features_step(steps_data):
    data = pd.DataFrame(
        {
            "Delivery_person_Age": [26, 29,25, 35],
            "Delivery_person_Ratings": [3.5, 4.0,4.5, 3.2],
            "Distance(km)": [11, 15, 10, 7],
            "Type_of_order": ["Buffet", "Drinks","Meal", "Snack"],
            "Type_of_vehicle": ["electric_scooter", "motorcycle", "motorcyle", "scooter"],
            "Time_taken(min)": [15, 25, 20, 18],
        }
    )

    features = [
        "Delivery_person_Age",
        "Delivery_person_Ratings",
        "Distance(km)",
        'Type_of_order_Buffet ', 
        'Type_of_order_Drinks ',
       'Type_of_order_Meal ', 
       'Type_of_order_Snack ',
       'Type_of_vehicle_electric_scooter ', 
       'Type_of_vehicle_motorcycle ',
       'Type_of_vehicle_scooter '
    ]
    target = "Time_taken(min)"
    y, X = get_features(target, features, data)


    schema = DataFrameSchema(
        {
            "Delivery_person_Age": Column(float, Check.isin([0.0, 100.0])),
            "Delivery_person_Ratings": Column(float, Check.isin([0.0, 5.0])),
            "Distance(km)": Column(float, Check.isin([1.0, 2000.0])),
            "Type_of_order_Buffet ": Column(float, Check.isin([0.0, 1.0])),
            "Type_of_order_Drinks ": Column(float, Check.isin([0.0, 1.0])),
            "Type_of_order_Meal ": Column(float, Check.isin([0.0, 1.0])),
            "Type_of_order_Snack ": Column(float, Check.isin([0.0, 1.0])),
            "Type_of_vehicle_electric_scooter ": Column(float, Check.isin([0.0, 1.0])),
            "Type_of_vehicle_motorcycle ": Column(float, Check.isin([0.0, 1.0])),
            "Type_of_vehicle_scooter ": Column(float, Check.isin([0.0, 1.0])),
        }
    )
    schema.validate(X)
    steps_data.X = X


def rename_columns_step(steps_data):
    processed_X = rename_columns(steps_data.X)
    assert list(processed_X.columns) == [
        "Delivery_person_Age",
        "Delivery_person_Ratings",
        "Distance(km)",
        "Type_of_order_Buffet",
        "Type_of_order_Drinks",
        "Type_of_order_Snack",
        "Type_of_order_Meal",
        "Type_of_vehicle_electric_scooter",
        "Type_of_vehicle_motorcycle",
        "Type_of_vehicle_scooter",
    ]