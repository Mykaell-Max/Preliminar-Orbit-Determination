import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from utils.coordinate_utils import convert_to_cartesian
from utils.ephemeris_utils import get_earth_position_and_velocity, solar_system_perturbations
from utils.orbital_mechanics import light_travel_correction, calculate_orbital_elements
from propagation.propagation import propagate_orbit
from .differential_correction import differential_correction
from utils.constants import MU_SUN
from utils.logging_config import setup_logger

logger = setup_logger(__name__)

def process_asteroid(name, observations, observatory=None):
   """
   Process the asteroid observations to determine its orbit
   """

   logger.header(f"\nCALCULANDO ÓRBITA DE {name} USANDO {len(observations)} OBSERVAÇÕES")

   t = [obs["timestamp"] for obs in observations]
   rho_hat = [convert_to_cartesian(obs["RA"], obs["DEC"]) for obs in observations]
   R, v_earth = get_earth_position_and_velocity(t, observatory)
   rho_initial = np.ones(len(t))

   logger.section("INICIANDO CORREÇÃO DIFERENCIAL")
   
   for iteration in range(10):
      logger.info(f"Iteração {iteration + 1}:")
      
      t_corr = light_travel_correction(rho_initial, t)
      R, v_earth = get_earth_position_and_velocity(t_corr)
      
      def perturbation_model(t, r, v):
         return solar_system_perturbations(t, r, v, t_corr[0])
      
      r_final, v_final = differential_correction(R, v_earth, rho_hat, t_corr, MU_SUN, perturbation_model)
      
      if r_final is None:
         logger.warning("Falha na convergência desta iteração. Tentando novamente...")
         rho_initial *= 0.7
         continue
      
      rho_new = np.array([np.linalg.norm(r_final - R[i]) for i in range(len(R))])
      delta_rho = np.max(np.abs(rho_new - rho_initial))
      
      logger.info(f"Δρ = {delta_rho:.6f} AU \n")

      if delta_rho < 1e-6:
         logger.info("Convergência alcançada!")
         break
      rho_initial = rho_new

   if r_final is None:
      logger.error("Falha ao encontrar órbita válida após todas as iterações.")
      return
   
   a, e, i, Omega, omega, theta = calculate_orbital_elements(r_final, v_final, MU_SUN)
   orbital_elements = {
      "a": f"{a:.6f} UA",
      "e": f"{e:.6f}",
      "i": f"{i:.6f}°",
      "Ω": f"{Omega:.6f}°",
      "ω": f"{omega:.6f}°",
      "θ": f"{theta:.6f}°"
   }

   logger.section("RESULTADOS")

   logger.result_box(f"ELEMENTOS ORBITAIS PARA {name}", orbital_elements)

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

   state_vector = {
      "Posição": f"{r_final} (|r| = {np.linalg.norm(r_final):.6f} AU)",
      "Velocidade": f"{v_final} (|v| = {np.linalg.norm(v_final):.6f} AU/dia)",
      "Resíduo RMS": f"{rms_residuo:.3f} segundos de arco"
   }
   
   logger.result_box("VETOR ESTADO FINAL", state_vector)
   
   return a, e, i, Omega, omega, theta, r_final, v_final, rms_residuo
