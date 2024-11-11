from enum import Enum, auto

class Convention(Enum):
    """
    Convention used for the provided data.

    NONE: No convention used
    
    OCEANIC: Oceanic convention - Direction to
    - Follows clockwise rotation from geographic North (0 degrees). East is in 90 degrees
    - Determines the direction towards which natural elements, e.g., waves/wind/currents, are moving to.
    
    
    METEOROLOGICAL: Meteorological or Nautical convention - Direction from
    - Follows clockwise rotation from geographic North (0 degrees). East is in 90 degrees
    - Determines the direction where natural elements, e.g., waves/wind/currents, are coming from.

    """
    NONE = auto()
    OCEANIC = auto()
    METEOROLOGICAL = auto()
