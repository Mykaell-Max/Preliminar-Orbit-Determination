import numpy as np
from datetime import timedelta
from utils.logging_config import setup_logger
from utils.orbital_mechanics import light_travel_correction
from utils.ephemeris_utils import get_earth_position_and_velocity, solar_system_perturbations
from models.observation import Observation
from models.orbit_solution import OrbitSolution
from core.differential_correction import differential_correction
from propagation.propagation import propagate_orbit
from utils.constants import MU_SUN

class OrbitDeterminer:
    """
    Coordinates the orbit determination process from observations to final solution.
    """
    def __init__(self, observations, object_name=None):
        """
        Initialize the orbit determination process
        
        Args:
            observations (list): List of Observation objects or dictionaries
            object_name (str, optional): Name of the observed object
        """

        self.observations = []
        for obs in observations:
            if isinstance(obs, dict):
                self.observations.append(Observation.from_dict(obs))
            else:
                self.observations.append(obs)
        
        self.object_name = object_name or "Unknown Object"
        self.logger = setup_logger(__name__)
        self.solution = None
        self._rho_estimates = None
        
    def determine_orbit(self, max_iterations=10, convergence_threshold=1e-6):
        """
        Perform the full orbit determination process
        
        Args:
            max_iterations (int): Maximum number of iterations for light-time correction
            convergence_threshold (float): Threshold for convergence in AU
            
        Returns:
            OrbitSolution: The calculated orbit solution or None if failed
        """
        self.logger.header(f"\nCALCULANDO ÓRBITA DE {self.object_name} USANDO {len(self.observations)} OBSERVAÇÕES")
        self.logger.section("INICIANDO CORREÇÃO DIFERENCIAL")
        
        t = [obs.timestamp for obs in self.observations]
        rho_hat = [obs.direction_vector for obs in self.observations]
        
        observatory = self.observations[0].observatory
        R, v_earth = get_earth_position_and_velocity(t, observatory)
        
        self._rho_estimates = np.ones(len(t))
        
        for iteration in range(max_iterations):
            self.logger.info(f"Iteração {iteration + 1}:")
            
            t_corr = light_travel_correction(self._rho_estimates, t)
            R, v_earth = get_earth_position_and_velocity(t_corr, observatory)
            
            def perturbation_model(t, r, v):
                return solar_system_perturbations(t, r, v, t_corr[0])
            
            r_final, v_final = differential_correction(R, v_earth, rho_hat, t_corr, MU_SUN, perturbation_model)
            
            if r_final is None:
                self.logger.warning("Falha na convergência desta iteração. Tentando novamente...")
                self._rho_estimates *= 0.7
                continue
            
            rho_new = np.array([np.linalg.norm(r_final - R[i]) for i in range(len(R))])
            delta_rho = np.max(np.abs(rho_new - self._rho_estimates))
            
            self.logger.info(f"Δρ = {delta_rho:.6f} AU \n")
            
            if delta_rho < convergence_threshold:
                self.logger.info("Convergência alcançada!")
                break
                
            self._rho_estimates = rho_new
        
        if r_final is not None:
            self.solution = OrbitSolution(
                r_vector=r_final,
                v_vector=v_final,
                epoch=t_corr[0],
                mu=MU_SUN
            )
            
            self._calculate_residuals(R, rho_hat, t_corr)
            
            self._log_results()
            
            return self.solution
        else:
            self.logger.error("Falha ao encontrar órbita válida após todas as iterações.")
            return None
    
    def _calculate_residuals(self, R, rho_hat, t_corr):
        """Calculate and store observation residuals"""
        def perturbation_model(t, r, v):
            return solar_system_perturbations(t, r, v, t_corr[0])
        
        r_pred, _ = propagate_orbit(
            self.solution.r_vector, 
            self.solution.v_vector, 
            t_corr, 
            t_corr[0], 
            MU_SUN, 
            perturbation_model
        )
        
        residuals_arc_sec = []
        
        for j in range(len(t_corr)):
            rho_vec = r_pred[j] - R[j]
            rho_norm = np.linalg.norm(rho_vec)
            rho_pred = rho_vec / rho_norm
            ang = np.arccos(np.clip(np.dot(rho_pred, rho_hat[j]), -1.0, 1.0))
            residuals_arc_sec.append(np.degrees(ang) * 3600)
        
        rms_residual = np.sqrt(np.mean(np.array(residuals_arc_sec)**2))
        
        self.solution.set_residuals({
            "rms_arcsec": rms_residual,
            "residuals_arcsec": residuals_arc_sec
        })
    
    def _log_results(self):
        """Log the orbit determination results"""
        if not self.solution:
            return
            
        self.logger.section("RESULTADOS")
        
        orbital_elements = self.solution.orbital_elements
        formatted_elements = {
            "a": f"{orbital_elements['a']:.6f} UA",
            "e": f"{orbital_elements['e']:.6f}",
            "i": f"{orbital_elements['i']:.6f}°",
            "Ω": f"{orbital_elements['Omega']:.6f}°",
            "ω": f"{orbital_elements['omega']:.6f}°",
            "θ": f"{orbital_elements['theta']:.6f}°"
        }
        
        self.logger.result_box(f"ELEMENTOS ORBITAIS PARA {self.object_name}", formatted_elements)
        
        r_mag = np.linalg.norm(self.solution.r_vector)
        v_mag = np.linalg.norm(self.solution.v_vector)
        
        state_vector = {
            "Posição": f"{self.solution.r_vector} (|r| = {r_mag:.6f} AU)",
            "Velocidade": f"{self.solution.v_vector} (|v| = {v_mag:.6f} AU/dia)",
            "Resíduo RMS": f"{self.solution.residuals['rms_arcsec']:.3f} segundos de arco"
        }
        
        self.logger.result_box("VETOR ESTADO FINAL", state_vector)