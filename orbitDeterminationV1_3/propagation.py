import numpy as np
from scipy.integrate import solve_ivp


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
