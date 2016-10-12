"""observation operator"""
import numpy as np
from . import (grid,
               horizontal)
from .vertical import Vertical2DInterpolator


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
