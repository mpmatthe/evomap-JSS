# read version from installed package
import sys
if sys.version.startswith('3.7'):
    from importlib_metadata import version
elif sys.version.startswith('3.8'):
    from importlib_metadata import version
else:
    from importlib.metadata import version
#__version__ = version("evomap")
__version__ = "0.1.0"