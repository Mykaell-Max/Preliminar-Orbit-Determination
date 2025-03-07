import numpy as np
from skyfield.api import load
from datetime import timedelta
from coordinate_utils import get_topocentric_correction
eph = load('de440.bsp')

def get_earth_position_and_velocity(times, observatory=None):
   """
   Enhanced function to get Earth positions efficiently
   """
   ts = load.timescale()
   sun = eph['sun']
   earth = eph['earth']
   t_years = np.array([t.year for t in times])
   t_months = np.array([t.month for t in times])
   t_days = np.array([t.day for t in times])
   t_hours = np.array([t.hour for t in times])
   t_minutes = np.array([t.minute for t in times])
   t_seconds = np.array([t.second + t.microsecond / 1e6 for t in times])
   t_sf = ts.utc(t_years, t_months, t_days, t_hours, t_minutes, t_seconds)
   
   # operaçao vetorizada
   earth_positions = earth.at(t_sf)
   sun_positions = sun.at(t_sf)
   positions = earth_positions.position.au - sun_positions.position.au
   velocities = earth_positions.velocity.au_per_d - sun_positions.velocity.au_per_d
   R = [positions[:, i] for i in range(positions.shape[1])]
   v = [velocities[:, i] for i in range(velocities.shape[1])]

   if observatory is not None:
      for i, t in enumerate(times):
         topo_offset = get_topocentric_correction(
               observatory['lat'], 
               observatory['lon'], 
               observatory['alt'], 
               t
         )
         gast = t_sf[i].gast * 15 
         gast_rad = np.radians(gast)
         sin_gast = np.sin(gast_rad)
         cos_gast = np.cos(gast_rad)
         rotated_offset = np.array([
               topo_offset[0] * cos_gast - topo_offset[1] * sin_gast,
               topo_offset[0] * sin_gast + topo_offset[1] * cos_gast,
               topo_offset[2]
         ])
         R[i] = R[i] + rotated_offset
         earth_rot_rate = 2 * np.pi / 86400  # rad/s
         earth_rot_rate_per_day = earth_rot_rate * 86400  # rad/d
         vel_correction = np.array([
               -earth_rot_rate_per_day * topo_offset[1],
               earth_rot_rate_per_day * topo_offset[0],
               0.0
         ])
         v[i] = v[i] + vel_correction
   return np.array(R), np.array(v)


def solar_system_perturbations(t, r, v, t_ref):
   """
   Calculate perturbation accelerations from major solar system bodies
   
   Args:
      t: Current time (days since t_ref)
      r: Position vector (AU)
      t_ref: Reference datetime
   
   Returns:
      np.array: Perturbing acceleration vector (AU/day^2)
   """
   current_time = t_ref + timedelta(days=float(t))
   ts = load.timescale()
   t_sf = ts.utc(current_time.year, current_time.month, current_time.day, current_time.hour, current_time.minute, current_time.second)
   
   sun = eph['sun']
   jupiter = eph['jupiter barycenter']
   saturn = eph['saturn barycenter']
   venus = eph['venus barycenter']
   earth = eph['earth']
   
   planets = [jupiter, saturn, venus, earth]
   planet_names = ['Jupiter', 'Saturn', 'Venus', 'Earth']
   planet_mus = [9.547e-4, 2.858e-4, 2.448e-6, 3.003e-6]  # AU^3/day^2
   a_pert = np.zeros(3)

   for planet, name, mu_planet in zip(planets, planet_names, planet_mus):
      planet_pos = planet.at(t_sf).position.au - sun.at(t_sf).position.au
      r_to_planet = planet_pos - r
      r_to_planet_norm = np.linalg.norm(r_to_planet)
      # perturbaçao direta
      a_direct = mu_planet * r_to_planet / r_to_planet_norm**3
      # perturbaçao indireta
      a_indirect = -mu_planet * planet_pos / np.linalg.norm(planet_pos)**3
      a_pert += a_direct + a_indirect
   return a_pert
