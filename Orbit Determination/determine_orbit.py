from models.orbit_determiner import OrbitDeterminer
from models.observation import Observation

def determine_orbit(object_name, observations, max_iterations=10, convergence_threshold=1e-6):
    """
    Modern, class-based entry point for orbit determination.
    
    Args:
        object_name (str): Name of the astronomical object
        observations (list): List of dictionaries or Observation objects
        max_iterations (int): Maximum number of iterations
        convergence_threshold (float): Convergence threshold
    
    Returns:
        tuple: Orbital elements (a, e, i, Omega, omega, theta) and state vectors (r, v)
              or None if determination fails
    """

    determiner = OrbitDeterminer(observations, object_name)
    
    solution = determiner.determine_orbit(
        max_iterations=max_iterations,
        convergence_threshold=convergence_threshold
    )
    
    if solution is None:
        return None, None, None, None, None, None, None, None, None
    
    elements = solution.orbital_elements
    
    return (
        elements['a'],
        elements['e'], 
        elements['i'],
        elements['Omega'],
        elements['omega'],
        elements['theta'],
        solution.r_vector,
        solution.v_vector,
        solution.residuals['rms_arcsec'] if solution.residuals else None
    )