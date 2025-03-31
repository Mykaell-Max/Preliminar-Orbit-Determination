import numpy as np
from skyfield.api import load

def convert_to_cartesian(ra, dec):
   """
   Converts equatorial coordinates (RA, DEC) to Cartesian coordinates (x, y, z)

   Args:
      ra (float): Right Ascension (degrees)
      dec (float): Declination (degrees)
   
   Returns:
      np.array: Cartesian coordinates
   """
   ra_rad = np.radians(ra)
   dec_rad = np.radians(dec)
   return np.array([np.cos(dec_rad) * np.cos(ra_rad), np.cos(dec_rad) * np.sin(ra_rad), np.sin(dec_rad)])


def get_topocentric_correction(lat, lon, alt, time):
    """
    Calculate topocentric correction vector for an observatory
    
    Args:
        lat (float): Observatory latitude in degrees
        lon (float): Observatory longitude in degrees
        alt (float): Observatory altitude in meters
        time (datetime): Time of observation
    
    Returns:
        np.array: Topocentric correction vector in AU
    """
    # raio da terra no equador (km)
    R_earth = 6378.137
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    ts = load.timescale()
    t_sf = ts.utc(time.year, time.month, time.day, time.hour, time.minute, time.second + time.microsecond / 1e6)
    earth_rotation = t_sf.gast * 15 
    earth_rotation_rad = np.radians(earth_rotation)

    # calcula a posiçao do observatorio 
    r_obs = np.array([
        (R_earth / 1000 + alt / 1000000) * np.cos(lat_rad) * np.cos(lon_rad + earth_rotation_rad),
        (R_earth / 1000 + alt / 1000000) * np.cos(lat_rad) * np.sin(lon_rad + earth_rotation_rad),
        (R_earth / 1000 + alt / 1000000) * np.sin(lat_rad)
    ])
    # conversao (certo dessa vez)
    r_obs_au = r_obs / 149597870.7
    return r_obs_au
