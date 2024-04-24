# Standard imports
import importlib.metadata

# Get the version number
# NOTE: poetry install required to bump version
__version__ = importlib.metadata.version("twinlab")
