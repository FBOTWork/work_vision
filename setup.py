## ! DO NOT MANUALLY INVOKE THIS setup.py, USE CATKIN INSTEAD

from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

# fetch values from package.xml
setup_args = generate_distutils_setup(
    packages=['work_behavior', 'work_behavior.states', 'work_behavior.machines', 'work_behavior.language_planning'],
    package_dir={'': 'src'},
    install_requires=[
        'langchain',
        'openai'
        'pyautogen'
    ])

setup(**setup_args)