from .interface import load_all

# Note: dashboard.py is a Dash app entry point, not imported here on
# purpose -- importing the package (e.g. for load_all) must not require
# Dash or pull in the app.
__all__ = ["load_all"]
