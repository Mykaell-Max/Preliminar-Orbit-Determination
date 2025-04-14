import numpy as np
from utils.constants import MU_SUN
from utils.orbital_mechanics import calculate_orbital_elements
from propagation.propagation import propagate_orbit

class OrbitSolution:
    """
    Represents the orbit determination solution, with state vectors and orbital elements.
    """
    def __init__(self, r_vector=None, v_vector=None, epoch=None, mu=MU_SUN):
        """
        Initialize an orbit solution
        
        Args:
            r_vector (np.array): Position vector in AU
            v_vector (np.array): Velocity vector in AU/day
            epoch (datetime): Epoch for the orbital elements
            mu (float): Gravitational parameter (default: Sun)
        """
        self.r_vector = r_vector
        self.v_vector = v_vector
        self.epoch = epoch
        self.mu = mu
        self._orbital_elements = None
        self._residuals = None
        
    @property
    def orbital_elements(self):
        """
        Calculate and cache the orbital elements
        
        Returns:
            dict: Dictionary containing a, e, i, Omega, omega, theta
        """
        if self._orbital_elements is None and self.r_vector is not None and self.v_vector is not None:
            a, e, inc, Omega, omega, theta = calculate_orbital_elements(
                self.r_vector, self.v_vector, self.mu)
            self._orbital_elements = {
                "a": a,
                "e": e,
                "i": inc,
                "Omega": Omega,
                "omega": omega,
                "theta": theta
            }
        return self._orbital_elements
    
    def propagate_to(self, target_times):
        """
        Propagate the orbit to specified times
        
        Args:
            target_times (list): List of datetime objects to propagate to
        
        Returns:
            tuple: Lists of position and velocity vectors at target times
        """
        if self.r_vector is None or self.v_vector is None or self.epoch is None:
            raise ValueError("Cannot propagate orbit without complete state vectors and epoch")
            
        r_results, v_results = propagate_orbit(
            self.r_vector, self.v_vector, target_times, self.epoch, self.mu)
        return r_results, v_results
    
    def set_residuals(self, residuals):
        """
        Set the residuals from the orbit determination process
        
        Args:
            residuals (dict): Dictionary with residual information
        """
        self._residuals = residuals
        
    @property
    def residuals(self):
        """Get the residuals of the orbit determination"""
        return self._residuals
    
    def to_dict(self):
        """
        Convert the orbit solution to a dictionary
        
        Returns:
            dict: Dictionary representation of the orbit solution
        """
        result = {
            "state_vector": {
                "position": self.r_vector.tolist() if self.r_vector is not None else None,
                "velocity": self.v_vector.tolist() if self.v_vector is not None else None,
                "epoch": self.epoch
            },
            "orbital_elements": self.orbital_elements
        }
        if self._residuals:
            result["residuals"] = self._residuals
        return result