import numpy as np
from datetime import timedelta
from .constants import C_LIGHT

def equations_of_motion(t, y, mu):
   """
   Returns the equations of motion for the two-body problem

   Args:
      t (float): Time
      y (np.array): State vector (position, velocity)
      mu (float): Gravitational parameter

   Returns:
      np.array: Derivative of the state vector
   """
   r = y[:3]
   v = y[3:]
   r_norm = np.linalg.norm(r)
   a = -mu * r / r_norm**3
   return np.hstack((v, a))


def calculate_orbital_elements(r, v, mu):
   """
   Calculates the orbital elements of an object given its position and velocity vectors

   Args:
      r (np.array): Position vector
      v (np.array): Velocity vector
      mu (float): Gravitational parameter
   
   Returns:
      float: Semi-major axis
      float: Eccentricity
      float: Inclination
      float: Longitude of Ascending Node
      float: Argument of Periapsis
      float: True Anomaly
   """
   r_norm = np.linalg.norm(r)
   v_norm = np.linalg.norm(v)
   energy = v_norm**2 / 2 - mu / r_norm
   a = -mu / (2 * energy) if energy < 0 else float('inf')
   h = np.cross(r, v)
   h_norm = np.linalg.norm(h)
   if h_norm < 1e-10:
      return float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf')
   e_vec = np.cross(v, h) / mu - r / r_norm
   e = np.linalg.norm(e_vec)
   i = np.degrees(np.arccos(h[2] / h_norm))
   n = np.array([-h[1], h[0], 0])
   n_norm = np.linalg.norm(n)
   if n_norm < 1e-10:
      Omega = 0.0
   else:
      Omega = np.degrees(np.arccos(n[0] / n_norm))
      if n[1] < 0:
         Omega = 360.0 - Omega
   if e < 1e-10:
      omega = 0.0
   else:
      omega = np.degrees(np.arccos(np.dot(n, e_vec) / (n_norm * e)))
      if e_vec[2] < 0:
         omega = 360.0 - omega
   v_dot_r = np.dot(v, r)
   theta = np.degrees(np.arccos(np.dot(r, e_vec) / (r_norm * e)))
   if v_dot_r < 0:
      theta = 360.0 - theta
   return a, e, i, Omega, omega, theta


def light_travel_correction(rho, t, c=C_LIGHT):
   """
   Applies light travel time correction to the observation times
   """
   tau = rho / c 
   tau_seconds = tau * 86400
   t_corr = [ti - timedelta(seconds=tau_seconds[i]) for i, ti in enumerate(t)]
   return t_corr


def estimate_angular_velocity(rho_hat, t):
   """
   Estimates the angular velocity of an object given its position vector and time

   Args:
      rho_hat (list): List of position vectors
      t (list): List of datetime objects

   Returns:
      np.array: Angular velocity vector
   """
   rho_hat = np.array(rho_hat)
   dt = np.array([(t[i] - t[0]).total_seconds() / 86400 for i in range(len(t))])
   A = np.column_stack((np.ones_like(dt), dt))
   rho_dot = []
   for i in range(3): 
      coeffs = np.linalg.lstsq(A, rho_hat[:, i], rcond=None)[0]
      rho_dot.append(coeffs[1])
   return np.array(rho_dot)
