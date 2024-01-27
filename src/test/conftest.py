import base64
import logging
import os
import sys

import pandas as pd
import pytest
import pytest_html

from .utils import load_data

DATA_FILE = "/Users/pmahon/Research/Dynamics/MITACS/src/main/resources/data/IEEE/pickles/main.pickle"

# Configure logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger()

# Set up directories for reports and plotting
logger.info("Setting up image directory for report plotting...")
os.environ['PYTEST_REPORT_IMAGES'] = os.path.dirname(os.path.realpath(__file__)) + "/report/images/"
os.makedirs(os.environ['PYTEST_REPORT_IMAGES'], exist_ok=True)

if os.listdir(os.environ['PYTEST_REPORT_IMAGES']):
    logger.info("Removing old images...")
    for f in os.listdir(os.environ['PYTEST_REPORT_IMAGES']):
        os.remove(os.path.join(os.environ['PYTEST_REPORT_IMAGES'], f))


@pytest.fixture(scope="module")
def training_set_indices():
    return {
        'library_index': pd.DatetimeIndex(pd.date_range(
            start="2020-09-01",
            end="2021-01-01 06:00:00",
            freq='H')),
        'prediction_index': pd.DatetimeIndex(pd.date_range(
            start="2021-01-01 07:00:00",
            end="2021-01-15 06:00:00",
            freq='H'))
    }


# todo: fix time splits
@pytest.fixture(scope="module")
def testing_set_indices():
    return {
        'library_index': pd.DatetimeIndex(pd.date_range(
            start="2020-09-01",
            end="2021-01-01 06:00:00",
            freq='H')),
        'training_index': pd.DatetimeIndex(pd.date_range(
            start="2021-01-01 07:00:00",
            end="2021-01-04 06:00:00",
            freq='H')),
        'prediction_index': pd.DatetimeIndex(pd.date_range(
            start="2020-12-18 07:00:00",
            end="2021-01-01 06:00:00",
            freq='H'))
    }


@pytest.fixture(scope="module")
def competition_set_indices():
    return {
        'library_index': pd.DatetimeIndex(pd.date_range(
            start="2020-09-01",
            end="2021-01-01 06:00:00",
            freq='H')),
        'training_index': pd.DatetimeIndex(pd.date_range(
            start="2021-01-01 07:00:00",
            end="2021-01-15 06:00:00",
            freq='H')),
        'prediction_index': pd.DatetimeIndex(pd.date_range(
            start="2021-01-18 07:00:00",
            end="2021-02-01 06:00:00",
            freq='H'))
    }


# noinspection DuplicatedCode
@pytest.fixture(scope="module")
def training_embedding(training_set_indices):
    """
    Defines the training set for the IEEE competition data via an embedding (not-compiled)

    :return Embedding: an embedding with library times defined, observers set to a 'target' variable to forecast.
    """

    # Load data
    return load_data(
        data_file=DATA_FILE,
        target='Load_(kW)',
        library_times=training_set_indices['library_index']
    )


@pytest.fixture(scope="module")
def testing_embedding(testing_set_indices):
    """
    Defines the test set for the IEEE competition data via an embedding (not-compiled)

    :return Embedding: an embedding with library times defined, observers set to a 'target' variable to forecast.
    """
    # Load data
    return load_data(
        data_file=DATA_FILE,
        target='Load_(kW)',
        library_times=testing_set_indices['library_index']
    )


@pytest.fixture(scope="module")
def competition_embedding(competition_set_indices):
    """
    Defines the validation set for the IEEE competition data via an embedding (not-compiled)

    :return Embedding: an embedding with library times defined, observers set to a 'target' variable to forecast.
    """
    # Load data
    return load_data(
        data_file=DATA_FILE,
        target='Load_(kW)',
        library_times=competition_set_indices['library_index']
    )


@pytest.fixture(scope="module")
def which_day_type():
    # Masking function to classify day types.
    #   Monday 7AM - Saturday 6AM -> 0
    #   Saturday 7AM - Sunday 6AM -> 1
    #   Sunday 7AM - Monday 6AM -> 2
    def label_time_stamp(timestamp: pd.Timestamp):

        # Filter out going forwards from Monday

        # Monday mornings are 'Sundays' -> 2
        if timestamp.dayofweek == 0 and timestamp.hour <= 6:
            return 2
        # Anything from between Monday mornings and Saturday is a 'Weekday' -> 0
        elif timestamp.dayofweek <= 4:
            return 0
        # Anything on Saturday morning is a 'Weekday'
        elif timestamp.dayofweek == 5 and timestamp.hour <= 6:
            return 1
        # Anything else on Saturday is a 'Saturday'
        elif timestamp.dayofweek == 5:
            return 1
        # Anything on a Sunday morning is a 'Saturday'
        elif timestamp.dayofweek == 6 and timestamp.hour <= 6:
            return 1
        # Anything else on a Sunday is a 'Sunday'
        else:
            return 2

    return label_time_stamp


@pytest.fixture(scope="module")
def day_types():
    return {0: 'Weekdays', 1: 'Saturdays', 2: 'Sundays'}


# Define a fixture to initialize image paths
@pytest.fixture
def image_paths(request):
    # Initialize an empty list for image paths
    request.node.image_paths = []

    # Return the list, so it can be manipulated within the test
    return request.node.image_paths


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item):
    outcome = yield
    report = outcome.get_result()

    # Correctly initialize 'extras' if it's not already an attribute of report
    if not hasattr(report, 'extras'):
        report.extras = []

    # Assuming you have set the 'image_paths' attribute in your test
    for image_path in getattr(item, 'image_paths', []):
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
            html = f'<div><img src="data:image/png;base64,{encoded_string}" alt="image" style="width:1200px; height:auto;"/></div>'
            report.extras.append(pytest_html.extras.image(encoded_string, mime_type='image/png'))
