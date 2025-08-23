"""
GeoAp - Gravitational Physics Analysis Package
"""

from src.utils.device_setup import setup_device

setup_device()


__version__ = "0.0.1"

from pathlib import Path

# Get project root (/GeoAP)
__PROJECT_ROOT__ = Path(__file__).parent
