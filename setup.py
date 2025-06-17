## ! DO NOT MANUALLY INVOKE THIS setup.py, USE CATKIN INSTEAD

from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

# Este setup.py declara os módulos pertencentes ao pacote work_vision.
# Se você tiver código python em "src/work_vision", ele será encontrado.
setup_args = generate_distutils_setup(
    packages=['work_vision'],
    package_dir={'': 'src'},
    install_requires=[
        'langchain',
        'openai',
        'pyautogen'
    ])

setup(**setup_args)