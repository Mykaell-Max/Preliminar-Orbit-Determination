from utils.coordinate_utils import convert_to_cartesian

class Observation:
    """
    Represents a single astronomical observation with equatorial coordinates and timestam.
    """
    def __init__(self, RA, DEC, timestamp, observatory=None):
        """
        Initialize an observation
        
        Args:
            RA (float): Right Ascension in degrees
            DEC (float): Declination in degrees
            timestamp (datetime): Time of observation
            observatory (dict, optional): Observatory information including lat, lon, alt
        """
        self.ra = RA
        self.dec = DEC
        self.timestamp = timestamp
        self.observatory = observatory
        self._direction_vector = None
    
    @property
    def direction_vector(self):
        """
        Calculate and cache the unit direction vector pointing to the observed object
        
        Returns:
            np.array: Unit direction vector
        """
        if self._direction_vector is None:
            self._direction_vector = convert_to_cartesian(self.ra, self.dec)
        return self._direction_vector
    
    @classmethod
    def from_dict(cls, data):
        """
        Create an Observation instance from a dictionary
        
        Args:
            data (dict): Dictionary containing RA, DEC, and timestamp
        
        Returns:
            Observation: New observation instance
        """
        return cls(
            RA=data["RA"],
            DEC=data["DEC"],
            timestamp=data["timestamp"],
            observatory=data.get("observatory")
        )
    
    def to_dict(self):
        """
        Convert the observation to a dictionary
        
        Returns:
            dict: Dictionary representation of the observation
        """
        result = {
            "RA": self.ra,
            "DEC": self.dec,
            "timestamp": self.timestamp
        }
        if self.observatory:
            result["observatory"] = self.observatory
        return result