"""observation operator"""
from . import (horizontal,
               vertical,
               exceptions)
from .vertical import Vertical2DInterpolator


class Operator(object):
    """Observation operator maps model values to observation locations

    Performs a horizontal interpolation followed by a vertical
    interpolation if needed.

    Ocean model horizontal layout can be specified via the `layout` keyword
    argument. By default it is set to 'regular', indicating regular lon/lat
    grids. But it may be changed to 'tripolar' for orca family grids or
    'regional' for models with complicated boundaries or longitude/latitude
    specifications.

    :param grid_longitudes: 2D array
    :param grid_latitudes: 2D array
    :param observed_longitudes: 1D array
    :param observed_latitudes: 1D array
    :param grid_depths: 1D/3D array representing either Z-levels or S-levels
    :param observed_depths: 1D array
    :param layout: ocean model horizontal layout, one of 'tripolar', 'regional'
                   or 'regular'
    :param has_halo: logical indicating tri-polar grid with halo
    """
    def __init__(self,
                 grid_longitudes,
                 grid_latitudes,
                 observed_longitudes,
                 observed_latitudes,
                 grid_depths=None,
                 observed_depths=None,
                 layout="regular",
                 has_halo=False):
        self.grid_depths = grid_depths
        self.observed_depths = observed_depths
        if layout.lower() == "tripolar":
            self.horizontal = horizontal.Tripolar(grid_longitudes,
                                                  grid_latitudes,
                                                  observed_longitudes,
                                                  observed_latitudes,
                                                  has_halo=has_halo)
        elif layout.lower() == "regular":
            self.horizontal = horizontal.Regular(grid_longitudes,
                                                 grid_latitudes,
                                                 observed_longitudes,
                                                 observed_latitudes)
        elif layout.lower() == "regional":
            self.horizontal = horizontal.Regional(grid_longitudes,
                                                  grid_latitudes,
                                                  observed_longitudes,
                                                  observed_latitudes)
        else:
            message = "unknown layout: {}".format(layout)
            raise exceptions.UnknownLayout(message)

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
        self.model_longitudes = model_longitudes
        self.model_latitudes = model_latitudes
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
        return horizontal.Regular(self.model_longitudes,
                                  self.model_latitudes,
                                  observed_longitudes,
                                  observed_latitudes).interpolate(model_field)

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
