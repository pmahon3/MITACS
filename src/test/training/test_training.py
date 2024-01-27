import logging
import os
import uuid

import matplotlib.pyplot as plt
import pytest

from ray.util.multiprocessing import Pool

from src.test.utils import fit_nonlinearity, fit_dimensionality, fit_greedy_nearest_neighbours
from edynamics.modelling_tools import KNearestNeighbours

logger = logging.getLogger()

compute_pool = Pool()

@pytest.mark.parametrize('variable', ['Load_(kW)'])
class TestTraining:

    def test_plot_data_set(self,
                           training_embedding,
                           variable,
                           training_set_indices,
                           image_paths
                           ):
        logger.info(f"Plotting {variable}...")
        # Plot the data series
        unique_id = uuid.uuid4()
        figure_path = os.path.join(os.getenv('PYTEST_REPORT_IMAGES'), f'{variable}_{unique_id}.png')

        plt.plot(training_embedding.data.loc[training_set_indices['library_index']][variable])
        plt.xticks(rotation=35)
        plt.ylabel(variable)
        plt.xlabel('Date')
        plt.savefig(figure_path)

        logger.info(f"Plot saved at {figure_path}")

        image_paths.append(figure_path)

        assert 1

    @pytest.mark.parametrize('max_dimensions', [10])
    def test_fitting(self,
                     training_set_indices,
                     training_embedding,
                     variable,
                     day_types,
                     which_day_type,
                     max_dimensions,
                     image_paths
                     ):
    #
    #    logger.info(f'Fitting dimensions using k nearest neighbours projection...')
    #    max_Es, figure_paths = fit_dimensionality(
    #        fitting_set_index=training_set_indices['library_index'],
    #        training_embedding=training_embedding, variable=variable,
    #        types=day_types, which_type=which_day_type,
    #        max_dimensions=max_dimensions, steps=1, step_size=1)
    #
    #    logger.info('\nBest Es:\n' + str(max_Es))
    #
    #    for path in figure_paths:
    #        image_paths.append(path)
    #
    #    logger.info(f'Fitting kernel for day ahead weighted least squares projection...')
    #    thetas, figure_paths = fit_nonlinearity(training_set_indices=training_set_indices['library_index'],
    #                                            training_embedding=training_embedding, variable=variable,
    #                                            types=day_types, which_type=which_day_type,
    #                                            dimensions=max_Es, steps=24, step_size=1)
    #
    #    logger.info('\nBest Thetas:\n' + str(thetas))
    #
    #    for path in figure_paths:
    #        image_paths.append(path)

        logger.info(f'Fitting observers for day ahead k-nearest neighbours projection using greedy nearest neighbours '
                    f'optimization...')

        performance = fit_greedy_nearest_neighbours(
            training_embedding=training_embedding,
            target=variable,
            projector=KNearestNeighbours(),
            times=training_set_indices['library_index'],
            max_dimensions=10,
            steps=1,
            step_size=1,
            compute_pool=compute_pool,
            improvement_threshold=0.0,
            verbose=True
        )

        performance = fit_greedy_nearest_neighbours(
            training_embedding=training_embedding,
            target=variable,
            projector=KNearestNeighbours(),
            times=training_set_indices['library_index'],
            max_dimensions=48,
            steps=24,
            step_size=1,
            compute_pool=compute_pool,
            improvement_threshold=0.0,
            verbose=True
        )

        assert 1
