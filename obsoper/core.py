"""observation operator"""
import numpy as np
from . import (grid,
               horizontal,
               vertical)
from .vertical import Vertical2DInterpolator


class Operator(object):
    """Observation operator maps model values to observation locations

    Performs a horizontal interpolation followed by a vertical
    interpolation if needed.

    .. note:: search methods and boundary definitions may result in
              non-convergent interpolation algorithms.
              See :class:`obsoper.horizontal.Horizontal` for more detail.

    :param grid_longitudes: 2D array
    :param grid_latitudes: 2D array
    :param observed_longitudes: 1D array
    :param observed_latitudes: 1D array
    :param grid_depths: 1D/3D array representing either Z-levels or S-levels
    :param observed_depths: 1D array
    :param has_halo: logical indicating tri-polar grid with halo
    :param search: grid search algorithm
    :param boundary: domain boundary description
    """
    def __init__(self,
                 grid_longitudes,
                 grid_latitudes,
                 observed_longitudes,
                 observed_latitudes,
                 grid_depths=None,
                 observed_depths=None,
                 has_halo=False,
                 search="cartesian",
                 boundary="polygon"):
        self.grid_depths = grid_depths
        self.observed_depths = observed_depths
        self.horizontal = horizontal.Horizontal(grid_longitudes,
                                                grid_latitudes,
                                                observed_longitudes,
                                                observed_latitudes,
                                                has_halo=has_halo,
                                                search=search,
                                                boundary=boundary)

    def interpolate(self, field):
        """Interpolates model field to observed locations

        :param field: array shaped (X, Y[, Z])
        :returns: array shapes (N[, L]) where N is number of observations
                  and L is the number of observed levels
        """
        if self.observed_depths is None:
            return self.horizontal.interpolate(field)
        section = vertical.Section(self.horizontal.interpolate(field),
                                   self.horizontal.interpolate(self.grid_depths))
        return section.interpolate(self.observed_depths)


class ObservationOperator(object):
    """Observation operator maps model values to observation locations

    Performs a horizontal interpolation followed by a vertical
    interpolation if needed.

    :param model_longitudes: 1D array
    :param model_latitudes: 1D array
    :param model_depths: 3D array
    """
    def __init__(self, model_longitudes, model_latitudes, model_depths=None):
        self.grid = grid.Regular2DGrid(model_longitudes, model_latitudes)
        self.model_depths = model_depths

    def interpolate(self, model_field, observed_longitudes,
                    observed_latitudes, observed_depths=None):
        """Interpolates model field to observed locations

        The convention is to specify coordinates in the order
        longitude, latitude.

        :param model_field: 2D/3D model array
        :param observed_longitudes: 1D array
        :param observed_latitudes: 1D array
        :param observed_depths: 3D array
        :returns: either 1D vector or 2D section of model_field in
                  observation space
        """
        # Horizontal interpolation
        model_section = self.horizontal_interpolate(model_field,
                                                    observed_longitudes,
                                                    observed_latitudes)
        if observed_depths is None:
            return model_section

        # Vertical interpolation
        depth_section = self.horizontal_interpolate(self.model_depths,
                                                    observed_longitudes,
                                                    observed_latitudes)
        return self.vertical_interpolate(model_section,
                                         depth_section,
                                         observed_depths)

    def horizontal_interpolate(self, model_field, observed_longitudes,
                               observed_latitudes):
        """interpolates model field to observed locations

        :param model_field: 2D/3D model array
        :param observed_longitudes: 1D array
        :param observed_latitudes: 1D array
        :returns: either 1D vector or 2D section of model_field in
                  observation space
        """
        # Detect observations inside grid
        mask = self.inside_grid(observed_longitudes, observed_latitudes)

        # np.ma.where is used to prevent masked elements being used as indices
        points = np.ma.where(mask)

        # Interpolate to observations inside grid
        search_result = self.grid.search(observed_longitudes[points],
                                         observed_latitudes[points])
        interpolator = horizontal.UnitSquare(*search_result)
        interpolated = interpolator(model_field)

        # Assemble result
        result = np.ma.masked_all(self.section_shape(observed_longitudes,
                                                     model_field))
        result[points] = interpolated
        return result

    @staticmethod
    def section_shape(positions, field):
        """defines shape of bilinear interpolated section/surface"""
        if field.ndim == 2:
            return (len(positions),)
        return (len(positions), field.shape[2])

    def inside_grid(self, observed_longitudes, observed_latitudes):
        """detect values inside model grid"""
        return self.grid.inside(observed_longitudes, observed_latitudes)

    @staticmethod
    def vertical_interpolate(model_section, model_depths,
                             observed_depths):
        """vertical interpolate model section to observed depths

        :param model_section: 2D array
        :param model_depths: 2D array
        :param observed_depths: 2D array
        :returns: model counterparts of observed depths
        """
        interpolator = Vertical2DInterpolator(model_depths, model_section)
        return interpolator(observed_depths)
