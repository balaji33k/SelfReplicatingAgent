import numpy as np

def geometry(solar_zenith, solar_azimuth, surface_tilt, surface_azimuth):
    """
    Calculate the angle of incidence of solar radiation on a tilted surface.

    Parameters
    ----------
    solar_zenith : float or array-like
        Solar zenith angle in degrees.
    solar_azimuth : float or array-like
        Solar azimuth angle in degrees.
    surface_tilt : float or array-like
        Surface tilt angle in degrees, where 0 is horizontal and 90 is vertical.
    surface_azimuth : float or array-like
        Surface azimuth angle in degrees.

    Returns
    -------
    angle_of_incidence : float or array-like
        Angle of incidence in degrees.
    """

    # Convert angles to radians
    solar_zenith_rad = np.radians(solar_zenith)
    solar_azimuth_rad = np.radians(solar_azimuth)
    surface_tilt_rad = np.radians(surface_tilt)
    surface_azimuth_rad = np.radians(surface_azimuth)

    # Calculate the cosine of the angle of incidence
    cos_aoi = (np.cos(solar_zenith_rad) * np.cos(surface_tilt_rad) +
               np.sin(solar_zenith_rad) * np.sin(surface_tilt_rad) *
               np.cos(solar_azimuth_rad - surface_azimuth_rad))

    # Clip the cosine of the angle of incidence to the range [-1, 1]
    cos_aoi = np.clip(cos_aoi, -1, 1)

    # Calculate the angle of incidence in degrees
    angle_of_incidence = np.degrees(np.arccos(cos_aoi))

    return angle_of_incidence


if __name__ == '__main__':
    # Test case 1: Negative solar zenith angle.  Should not return NaN.
    solar_zenith = -10
    solar_azimuth = 180
    surface_tilt = 30
    surface_azimuth = 180
    aoi = geometry(solar_zenith, solar_azimuth, surface_tilt, surface_azimuth)
    print(f"AoI with negative zenith: {aoi}")
    assert not np.isnan(aoi), "AoI should not be NaN with negative zenith."
    assert np.isclose(aoi, 130.0), "AoI calculation incorrect"

    # Test case 2: Zero tilt, arbitrary azimuth
    solar_zenith = 60
    solar_azimuth = 180
    surface_tilt = 0
    surface_azimuth = 90
    aoi = geometry(solar_zenith, solar_azimuth, surface_tilt, surface_azimuth)
    print(f"AoI with zero tilt: {aoi}")
    assert np.isclose(aoi, 60.0), "AoI calculation incorrect"

    # Test case 3: Zenith and tilt both 0
    solar_zenith = 0
    solar_azimuth = 0
    surface_tilt = 0
    surface_azimuth = 0
    aoi = geometry(solar_zenith, solar_azimuth, surface_tilt, surface_azimuth)
    print(f"AoI with zero zenith and tilt: {aoi}")
    assert np.isclose(aoi, 0.0), "AoI calculation incorrect"

    # Test case 4: All zeros
    solar_zenith = 0
    solar_azimuth = 0
    surface_tilt = 0
    surface_azimuth = 0
    aoi = geometry(solar_zenith, solar_azimuth, surface_tilt, surface_azimuth)
    print(f"AoI with all zeros: {aoi}")
    assert np.isclose(aoi, 0.0), "AoI calculation incorrect"

    # Test case 5: Extreme zenith and tilt (check clipping handles it)
    solar_zenith = 180
    solar_azimuth = 0
    surface_tilt = 90
    surface_azimuth = 0
    aoi = geometry(solar_zenith, solar_azimuth, surface_tilt, surface_azimuth)
    print(f"AoI with extreme zenith and tilt: {aoi}")
    assert np.isclose(aoi, 90.0), "AoI calculation incorrect"

    # Test case 6: Array input
    solar_zenith = np.array([-10, 0, 30])
    solar_azimuth = np.array([180, 90, 0])
    surface_tilt = 30
    surface_azimuth = 180
    aoi = geometry(solar_zenith, solar_azimuth, surface_tilt, surface_azimuth)
    print(f"AoI with array input: {aoi}")
    assert not np.any(np.isnan(aoi)), "AoI should not be NaN with array input."
    assert np.allclose(aoi, [130., 108.21407884, 81.67204731]), "AoI calculation incorrect for array input"

    print("All test cases passed!")