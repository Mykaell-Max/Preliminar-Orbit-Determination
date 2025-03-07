import numpy as np
from scipy.optimize import least_squares, minimize, differential_evolution
from scipy.integrate import solve_ivp
from datetime import timedelta
from skyfield.api import load
import warnings

# constantes 
MU_SUN = 0.000295912208  # AU^3/dia^2
C_LIGHT = 173.1446 # AU/dia
eph = load('de440.bsp')


# --------------------------------------------------------------------------------------------------------
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


# --------------------------------------------------------------------------------------------------------
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


# --------------------------------------------------------------------------------------------------------
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


# --------------------------------------------------------------------------------------------------------
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


# --------------------------------------------------------------------------------------------------------
def propagate_orbit(r0, v0, t, t_ref, mu, perturbations=None):
   """
   Enhanced orbit propagation with optional perturbation forces
   
   Args:
      r0, v0: Initial state vectors
      t: List of datetime objects
      t_ref: Reference time
      mu: Gravitational parameter
      perturbations: Optional function for additional accelerations
   """
   t_days = np.array([(ti - t_ref).total_seconds() / 86400 for ti in t])
   t_span = (min(t_days), max(t_days)) if len(t_days) > 1 else (0, t_days[0] if t_days[0] > 0 else 0.001)
   y0 = np.hstack((r0, v0))
   def equations_with_perturbations(t, y, mu): # adicionei perturbação
      r = y[:3]
      v = y[3:]
      r_norm = np.linalg.norm(r)
      a = -mu * r / r_norm**3
      if perturbations is not None:
         a_pert = perturbations(t, r, v)
         a = a + a_pert   
      return np.hstack((v, a))
   sol = solve_ivp(equations_with_perturbations, t_span, y0, t_eval=t_days, 
      method='DOP853',  # testando um metodo de ordem maior
      rtol=1e-12, 
      atol=1e-14, 
      args=(mu,)
   )
   if not sol.success:
      raise ValueError("Integration failed: " + sol.message)
   return sol.y[:3].T, sol.y[3:].T


# --------------------------------------------------------------------------------------------------------
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


# --------------------------------------------------------------------------------------------------------
def light_travel_correction(rho, t, c=C_LIGHT):
   """
   Applies light travel time correction to the observation times
   """
   tau = rho / c 
   tau_seconds = tau * 86400
   t_corr = [ti - timedelta(seconds=tau_seconds[i]) for i, ti in enumerate(t)]
   return t_corr


# --------------------------------------------------------------------------------------------------------
def calculate_f_and_g(dt, r0, mu):
   """
   Calculates the f and g functions for the universal variable formulation

   Args:
      dt (float): Time since the initial observation
      r0 (np.array): Initial position vector
      mu (float): Gravitational parameter
   
   Returns:
      float: f function
      float: g function
   """
   r0_norm = np.linalg.norm(r0)
   a0 = -mu * r0 / r0_norm**3
   f = 1.0 - (mu / (2 * r0_norm**3)) * dt**2 + (mu / (24 * r0_norm**3)) * dt**4
   g = dt - (mu / (6 * r0_norm**3)) * dt**3 + (mu**2 / (120 * r0_norm**6)) * dt**5
   if abs(dt) > 1.0:
      v_circ = np.sqrt(mu / r0_norm)
      u = dt * v_circ / r0_norm
      for _ in range(5):
         c = np.cos(u)
         s = np.sin(u)
         f_u = r0_norm * c + np.dot(r0, a0) * s / v_circ - r0_norm
         f_u_prime = -r0_norm * s + np.dot(r0, a0) * c / v_circ
         delta_u = -f_u / f_u_prime if abs(f_u_prime) > 1e-10 else 0
         u += delta_u
         if abs(delta_u) < 1e-10:
               break
      f = 1.0 - (mu / r0_norm) * (1.0 - np.cos(u))
      g = dt - np.sqrt(r0_norm**3 / mu) * (u - np.sin(u))
   return f, g


# --------------------------------------------------------------------------------------------------------
def iod_initial_estimate(R, rho_hat, t):
   """
   Estimates the initial position and velocity of an object using the method of initial osculating elements

   Args:
      R (list): List of position vectors (earth)
      rho_hat (list): List of unit direction vectors
      t (list): List of datetime objects
   """
   if len(R) == 4:
      R_select = np.array(R)
      rho_hat_select = np.array(rho_hat)
      t_select = t

   if len(R) >= 4: # para um caso mais geral, teste com mais de 4 observaççoes
      idx = [0, len(R) // 3, 2 * len(R) // 3, -1]
      R_select = np.array([R[i] for i in idx])
      rho_hat_select = np.array([rho_hat[i] for i in idx])
      t_select = [t[i] for i in idx]
      
   t0 = t_select[0]
   dt = [(ti - t0).total_seconds() / 86400 for ti in t_select]
   D = []
   for i in range(len(rho_hat_select) - 1):
      D.append(np.cross(rho_hat_select[i], rho_hat_select[i+1]))
   D_norms = [np.linalg.norm(d) for d in D]
   is_collinear = any(d_norm < 1e-10 for d_norm in D_norms)
   if is_collinear:
      print("Detectadas observações quase colineares. Usando método alternativo.") # caso de observações colineares, é um problema que tive em alguns testes
      ang_velocities = []
      for i in range(len(rho_hat) - 1):
         dot_prod = np.clip(np.dot(rho_hat[i], rho_hat[i+1]), -1.0, 1.0)
         ang = np.arccos(dot_prod)
         dt_days = (t[i+1] - t[i]).total_seconds() / 86400
         ang_velocities.append(ang / dt_days)
      avg_ang_velocity = np.mean(ang_velocities)
      
      # estimar distancia usando velocidade angular media
      # Para NEOs típicos a ~1AU com velocidade orbital ~30 km/s
      estimated_rho = 0.01 / avg_ang_velocity if avg_ang_velocity > 1e-10 else 1.5
      estimated_rho = np.clip(estimated_rho, 0.1, 5.0)
      r0 = R[0] + estimated_rho * rho_hat[0]
      dt1 = (t[1] - t[0]).total_seconds() / 86400
      rho_dot = (rho_hat[1] - rho_hat[0]) / dt1
      v0 = R[1] - R[0] + estimated_rho * rho_dot
      return r0, v0 / dt1
   
   # metodo classico para nao colineares
   A = np.zeros((len(rho_hat_select), len(rho_hat_select)))
   for i in range(len(rho_hat_select)):
      A[i, i] = 1.0
      for j in range(len(rho_hat_select)):
         if i != j:
               A[i, j] = -np.dot(rho_hat_select[i], rho_hat_select[j])
   
   # Vetor de termos independentes
   b = np.zeros(len(rho_hat_select))
   for i in range(len(rho_hat_select) - 1):
      b[i] = np.dot(R_select[i+1] - R_select[i], rho_hat_select[i])
   b[-1] = np.dot(R_select[0] - R_select[-1], rho_hat_select[-1])
   try:
      rho = np.linalg.solve(A, b)
      if any(r <= 0 or r > 10 for r in rho):
         print("Distâncias calculadas nao razoáveis.")
         return None, None
      r = np.zeros((len(rho_hat_select), 3))
      for i in range(len(rho_hat_select)):
         r[i] = R_select[i] + rho[i] * rho_hat_select[i]
      if len(dt) >= 3:
         y = r.copy()
         X = np.column_stack([np.ones(len(dt)), np.array(dt)])
         coeffs = np.zeros((3, 2))
         for i in range(3):
               coeffs[i], _, _, _ = np.linalg.lstsq(X, y[:, i], rcond=None)
         r0 = coeffs[:, 0]
         v0 = coeffs[:, 1]
      else:
         try: 
            # teste
            f1, g1 = calculate_f_and_g(dt[1], r[0], MU_SUN)
            f2, g2 = calculate_f_and_g(dt[-1], r[0], MU_SUN)
            A_v = np.array([[f1, g1], [f2, g2]])
            b_v = np.array([r[1], r[-1]])
            v0 = np.zeros(3)
            for i in range(3):
               sol = np.linalg.lstsq(A_v, [b_v[0][i], b_v[1][i]], rcond=None)[0]
               v0[i] = sol[1]
         except:
               v0 = (r[1] - r[0]) / dt[1]

      v_norm = np.linalg.norm(v0)
      if v_norm > 0.2:  # > 120 km/s é fisicamente improvável
         print(f"Velocidade inicial muito alta: {v_norm:.6f} AU/dia")
         v0 = v0 * (0.1 / v_norm)
      
      return r[0], v0
   
   except np.linalg.LinAlgError as e:
      print(f"Erro na estimativa: {e}")
      return None, None


# --------------------------------------------------------------------------------------------------------
def differential_correction(R, v_earth, rho_hat, t, mu_sun, perturbations=None):
   """
   Applies differential correction to refine the initial estimate of the object's orbit

   Args:
      R (list): List of position vectors (earth)
      v_earth (list): List of velocity vectors (earth)
      rho_hat (list): List of unit direction vectors
      t (list): List of datetime objects
      mu_sun (float): Gravitational parameter of the Sun
   
   Returns:
      np.array: Refined position vector
      np.array: Refined velocity vector
   """
   r_initial, v_initial = iod_initial_estimate(R, rho_hat, t)
   
   if r_initial is None:
      best_result = None
      best_cost = float('inf')
      r_earth_sun = np.linalg.norm(R[0])
      bounds = [
         # Posição: região dentro de 5 AU do Sol
         (-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0),
         # Velocidade: até 0.07
         (-0.07, 0.07), (-0.07, 0.07), (-0.07, 0.07)
      ]
      def objective(params):
         r0, v0 = params[:3], params[3:]
         r_norm = np.linalg.norm(r0)
         if r_norm < 0.1 or r_norm > 10.0:
               return 1e10
         
         try:
            r_pred, _ = propagate_orbit(r0, v0, t, t[0], mu_sun)
            cost = 0
            weights = np.linspace(1.0, 0.8, len(t))
            
            for i in range(len(t)):
               rho_vec = r_pred[i] - R[i]
               rho_norm = np.linalg.norm(rho_vec)
               if rho_norm < 1e-10:
                  return 1e10
               rho_pred = rho_vec / rho_norm
               ang_error = np.arccos(np.clip(np.dot(rho_pred, rho_hat[i]), -1.0, 1.0))
               cost += weights[i] * ang_error**2
            r_norm = np.linalg.norm(r0)
            v_norm = np.linalg.norm(v0)
            energy = v_norm**2 / 2 - mu_sun / r_norm
            h = np.cross(r0, v0)
            h_norm = np.linalg.norm(h)
            if h_norm > 1e-10:
               e_vec = np.cross(v0, h) / mu_sun - r0 / r_norm
               e = np.linalg.norm(e_vec)
               if energy > 0:  
                  cost += 200 * energy  
               elif -mu_sun / (2 * energy) > 5.0: 
                  cost += 50 * (-mu_sun / (2 * energy) - 5.0)
               if e > 0.95: 
                  cost += 100 * (e - 0.95)**2
            return cost
         except Exception:
               return 1e10
      for trial in range(7): 
         r_scale = 0.7 + 0.3 * trial 
         v_scale = 0.008 + 0.008 * trial
         x0 = np.zeros(6)
         random_offset = np.random.uniform(-0.05, 0.05, 3) if trial > 0 else np.zeros(3)
         x0[:3] = R[0] + r_scale * (rho_hat[0] + random_offset)
         rho_hat_dot = estimate_angular_velocity(rho_hat, t)
         v_offset = np.random.uniform(-0.003, 0.003, 3) if trial > 0 else np.zeros(3)
         x0[3:] = v_earth[0] + v_scale * rho_hat[0] + r_scale * rho_hat_dot + v_offset
         try:
            methods = ['Nelder-Mead', 'Powell'] if trial < 2 else ['Nelder-Mead']
            for method in methods:
               result = minimize(objective, x0, method=method, bounds=bounds, options={'maxiter': 1500})
               if result.fun < best_cost:
                  best_cost = result.fun
                  best_result = result.x
         except:
               continue
      if best_result is not None:
         r_initial, v_initial = best_result[:3], best_result[3:]
      else:
         # Se tudo falhar -> tentar otimização global 
         try:
            with warnings.catch_warnings():
               warnings.simplefilter("ignore")
               result = differential_evolution(objective, bounds, popsize=25, maxiter=40, strategy='best1bin', tol=1e-7)
            r_initial, v_initial = result.x[:3], result.x[3:]
         except:
               print("Falha na otimização global.")
               return None, None
   if r_initial is None:
      return None, None
   
   def residuals(params):
      r0, v0 = params[:3], params[3:]
      r_norm = np.linalg.norm(r0)
      v_norm = np.linalg.norm(v0)
      energy = 0.5 * v_norm**2 - mu_sun / r_norm
      h = np.cross(r0, v0)
      h_norm = np.linalg.norm(h)
      penalty = 0
      if h_norm > 1e-10:
         a = -mu_sun / (2 * energy) if energy < 0 else float('inf')
         e_vec = np.cross(v0, h) / mu_sun - r0 / r_norm
         e = np.linalg.norm(e_vec)
         if energy > 0:  # Órbita hiperbólica
            penalty += 500 * energy
         elif a < 0.3:  # Órbita muito pequena
            penalty += 100 * (0.3 - a)**2
         elif a > 5.0:  # Órbita muito grande
            penalty += 50 * (a - 5.0)**2
         if e > 0.98:  # Excentricidade muito alta
            penalty += 200 * (e - 0.98)**2
      else:
         penalty += 1000
      try:
         r_pred, _ = propagate_orbit(r0, v0, t, t[0], mu_sun, perturbations)
      except Exception as e:
         return np.ones(len(t) * 3) * 1e8 + penalty
      weights = np.linspace(1.2, 0.8, len(t))
      resid = []
      for i in range(len(t)):
         rho_vec = r_pred[i] - R[i]
         rho_norm = np.linalg.norm(rho_vec)
         if rho_norm < 1e-10:
               return np.ones(len(t) * 3) * 1e8 + penalty
         rho_pred = rho_vec / rho_norm
         resid.append((rho_pred - rho_hat[i]) * weights[i])
      flat_resid = np.concatenate(resid)
      if penalty > 0:
         flat_resid = flat_resid + penalty * np.ones_like(flat_resid) / len(flat_resid)
      return flat_resid
   
   # rfinamento final com least_squares usando bounds mais amplos
   initial_params = np.hstack((r_initial, v_initial))
   bounds = ([-15, -15, -15, -0.15, -0.15, -0.15], [15, 15, 15, 0.15, 0.15, 0.15])
   
   try:
      result = least_squares(residuals, initial_params, method='trf', bounds=bounds, ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=10000, verbose=0)
      r_final, v_final = result.x[:3], result.x[3:]
      r_norm = np.linalg.norm(r_final)
      v_norm = np.linalg.norm(v_final)
      energy = 0.5 * v_norm**2 - mu_sun / r_norm
      a = -mu_sun / (2 * energy) if energy < 0 else float('inf')
      h = np.cross(r_final, v_final)
      h_norm = np.linalg.norm(h)
      if h_norm > 1e-10:
         e_vec = np.cross(v_final, h) / mu_sun - r_final / r_norm
         e = np.linalg.norm(e_vec)
         i = np.degrees(np.arccos(abs(h[2]) / h_norm))
         if energy > 0: 
            print(f"Solução hiperbólica: energy={energy:.8f}, e={e:.6f}")
            if energy > 1e-4: 
               return None, None
         elif a > 10.0:
            print(f"Solução com semi-eixo maior excessivo: a={a:.6f} AU")
            return None, None
         elif e > 0.99:
            print(f"Solução com excentricidade excessiva: e={e:.6f}")
            return None, None
         print(f"Solução aceita: a={a:.6f} AU, e={e:.6f}, i={i:.6f}°")
         return r_final, v_final
      else:
         print("Solução com momento angular quase nulo.")
         return None, None
   except Exception as e:
      print(f"Erro no refinamento: {e}")
      return None, None


# --------------------------------------------------------------------------------------------------------
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


# --------------------------------------------------------------------------------------------------------
def process_asteroid(name, observations, observatory=None):
   """
   Processa observações de um asteroide e determina sua órbita
   """
   print(f"\n### Calculando órbita de {name} usando {len(observations)} observaçoes###")
   t = [obs["timestamp"] for obs in observations]
   rho_hat = [convert_to_cartesian(obs["RA"], obs["DEC"]) for obs in observations]
   R, v_earth = get_earth_position_and_velocity(t, observatory)
   rho_initial = np.ones(len(t))

   for iteration in range(10):
      print(f"Iteração {iteration + 1}:")
      t_corr = light_travel_correction(rho_initial, t)
      R, v_earth = get_earth_position_and_velocity(t_corr)
      def perturbation_model(t, r, v):
            return solar_system_perturbations(t, r, v, t_corr[0])
      r_final, v_final = differential_correction(R, v_earth, rho_hat, t_corr, MU_SUN, perturbation_model)
      if r_final is None:
         print("Falha na convergência desta iteração. Tentando novamente...")
         rho_initial *= 0.7
         continue
      rho_new = np.array([np.linalg.norm(r_final - R[i]) for i in range(len(R))])
      delta_rho = np.max(np.abs(rho_new - rho_initial))
      print(f"Δρ = {delta_rho:.6f} AU")
      if delta_rho < 1e-6:
         print("Convergência alcançada!")
         break
      rho_initial = rho_new
   if r_final is None:
      print("Falha ao encontrar órbita válida após todas as iterações.")
      return
   a, e, i, Omega, omega, theta = calculate_orbital_elements(r_final, v_final, MU_SUN)
   def perturbation_model(t, r, v):
      return solar_system_perturbations(t, r, v, t_corr[0])
   r_pred, _ = propagate_orbit(r_final, v_final, t_corr, t_corr[0], MU_SUN, perturbation_model)
   residuos_ang = []
   for j in range(len(t)):
      rho_vec = r_pred[j] - R[j]
      rho_norm = np.linalg.norm(rho_vec)
      rho_pred = rho_vec / rho_norm
      ang = np.arccos(np.clip(np.dot(rho_pred, rho_hat[j]), -1.0, 1.0))
      residuos_ang.append(np.degrees(ang) * 3600)
   rms_residuo = np.sqrt(np.mean(np.array(residuos_ang)**2))
   print(f"\nr_final = {r_final}, |r| = {np.linalg.norm(r_final):.6f} AU")
   print(f"v_final = {v_final}, |v| = {np.linalg.norm(v_final):.6f} AU/dia")
   print(f"Resíduo RMS: {rms_residuo:.3f} segundos de arco")

   print(f"\n### Elementos Orbitais Finais para ({name}) ###")
   print(f"a = {a:.6f} UA")
   print(f"e = {e:.6f}")
   print(f"i = {i:.6f}°")
   print(f"Ω = {Omega:.6f}°")
   print(f"ω = {omega:.6f}°")
   print(f"θ = {theta:.6f}°")

   
   if name == "Apophis":
      a_real = 0.9223
      e_real = 0.1911
      i_real = 3.33
      print("\n### Comparação com elementos reais (Apophis) ###")
      print(f"a real ≈ {a_real:.4f} UA, diferença: {abs(a - a_real):.4f} UA")
      print(f"e real ≈ {e_real:.4f}, diferença: {abs(e - e_real):.4f}")
      print(f"i real ≈ {i_real:.2f}°, diferença: {abs(i - i_real):.2f}°")
   
   elif name == "2023 DZ2":
      a_real = 2.082
      e_real = 0.5242
      i_real = 0.15
      print("\n### Comparação com elementos reais (2023 DZ2) ###")
      print(f"a real ≈ {a_real:.4f} UA, diferença: {abs(a - a_real):.4f} UA")
      print(f"e real ≈ {e_real:.4f}, diferença: {abs(e - e_real):.4f}")
      print(f"i real ≈ {i_real:.2f}°, diferença: {abs(i - i_real):.2f}°")
   

# --------------------------------------------------------------------------------------------------------
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
