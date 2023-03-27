from typing import Callable
import pandas as pd


def get_prediction_periods(mask_functions, indices: pd.MultiIndex, daily_prediction_start: pd.Timestamp, steps):
    for function in mask_functions:
        function_indices = indices[list(map(function, indices.get_level_values(level=len(indices.names))))]


# Day masks
def day_masks(time: pd.Timestamp) -> Callable:
    if 0 <= time.dayofweek <= 4:
        return lambda y: True if 0 <= y.dayofweek <= 4 else False
    elif time.dayofweek == 5:
        return lambda y: True if y.dayofweek == 5 else False
    elif time.dayofweek == 6:
        return lambda y: True if y.dayofweek == 6 else False
    else:
        raise ValueError('Masking Error')


# Start Times
def weekdays_starts(x: pd.Timestamp) -> bool:
    if 0 <= x.dayofweek <= 4:
        return x.hour == 7
    else:
        return False


def saturdays_starts(x: pd.Timestamp) -> bool:
    return x.dayofweek == 5 and x.hour == 7


def sundays_starts(x: pd.Timestamp) -> bool:
    return x.dayofweek == 6 and x.hour == 7
