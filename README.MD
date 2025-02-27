# Orbit Calculation Module

A Python module for calculating the orbital elements of astronomical objects based on astrometric observations.

## Overview

This module implements methods to determine an object's orbit in space based on observations from Earth. It uses the Gauss method and differential correction techniques to calculate the six Keplerian orbital elements that fully describe the object's trajectory around the Sun.

## Dependencies

- numpy
- scipy
- skyfield
- datetime

## Features

- Convert equatorial coordinates (RA, DEC) to Cartesian coordinates
- Retrieve Earth position and velocity at observation times
- Implement numerical integration of orbital motion
- Apply light-travel time corrections
- Perform initial orbit determination
- Apply differential correction to refine orbital solutions
- Calculate and report the six Keplerian orbital elements

## Input Data

The module expects the following input:

1. **Observation Data (Astrometry)**
   - Time of observation (UTC)
   - Right Ascension (RA) in degrees
   - Declination (DEC) in degrees

## Output

The module calculates the following orbital elements:

1. **Orbital Elements (Classical Keplerian)**
   - Semi-major axis (a): Defines the size of the orbit
   - Eccentricity (e): Defines how elongated the orbit is
   - Inclination (i): Tilt of the orbit relative to the ecliptic plane
   - Longitude of Ascending Node (Ω): Where the orbit crosses the reference plane
   - Argument of Periapsis (ω): Orientation of the ellipse within the orbit
   - True Anomaly (θ): Position of the object along the orbit at a given time

## Usage Example

```python
from orbit_calculation import process_asteroid
from datetime import datetime, timezone

# Example observations for asteroid "Apophis"
observations = [
   {"RA": 61.533666, "DEC": 16.917944, "timestamp": datetime(2004, 3, 15, 2, 35, 21, 696000, tzinfo=timezone.utc)},
   {"RA": 61.548958, "DEC": 16.920972, "timestamp": datetime(2004, 3, 15, 2, 58, 4, 320000, tzinfo=timezone.utc)},
   {"RA": 61.562083, "DEC": 16.923722, "timestamp": datetime(2004, 3, 15, 3, 17, 34, 272000, tzinfo=timezone.utc)},
   {"RA": 61.564917, "DEC": 16.925194, "timestamp": datetime(2004, 3, 15, 3, 21, 58, 848000, tzinfo=timezone.utc)},
]

# Process the observations and calculate orbital elements
process_asteroid("Apophis", observations)
```

## Algorithm Description

The module implements the following computational procedure:

1. Reads observation data (RA, DEC, timestamps)
2. Converts equatorial coordinates to Cartesian vectors
3. Retrieves Earth's position at observation times
4. Applies initial orbit determination using Gauss's method
5. Iteratively refines the solution using differential correction
6. Applies light-travel time correction
7. Calculates and reports the final orbital elements

## Handling Edge Cases

The module includes special handling for:
- Collinear observations
- Hyperbolic solutions
- Orbits with excessive eccentricity
- Failed convergence

## References

Orbit determination methods are based on:
- Gauss's method for initial orbit determination
- Differential correction for orbit refinement
- Two-body problem integration for orbit propagation