package_dir = \
{'': 'src'}

packages = \
['evomap',
 'evomap.data',
 'evomap.data.cars',
 'evomap.data.tnic_sample_small',
 'evomap.data.tnic_snapshot',
 'evomap.data.tnic_snapshot_small',
 'evomap.mapping',
 'evomap.mapping.evomap']

package_data = \
{'': ['*']}

install_requires = \
['Cython>=0.29.28',
 'ipykernel>=6.13.0',
 'matplotlib>=3.5.1',
 'numba>=0.55.1,<0.56.0',
 'numpy==1.21.6',
 'pandas>=1.3.3',
 'scipy>=1.7.0',
 'seaborn>=0.11.2',
 'statsmodels>=0.13.2,<0.14.0',
 'setuptools>=62.3.2']

setup_kwargs = {
    'name': 'evomap',
    'version': '0.1.0',
    'description': 'A Python Toolbox for Mapping Evolving Relationship Data',
    'long_description': open('README.md', 'r', encoding='utf-8').read(),
    'author': 'Maximilian Matthe, Daniel M. Ringel, and Bernd Skiera',
    'author_email': 'matthe@wiwi.uni-frankfurt.de',
    'maintainer': None,
    'maintainer_email': None,
    'url': None,
    'package_dir': package_dir,
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '>=3.9,<3.11',
}
from build import *
build(setup_kwargs)

from setuptools import setup
setup(**setup_kwargs)