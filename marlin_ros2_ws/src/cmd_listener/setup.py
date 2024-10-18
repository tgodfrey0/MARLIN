from setuptools import setup
import os
from glob import glob

package_name = 'cmd_listener'

setup(
    name = package_name,
    version = '0.0.0',
    packages = [package_name],
    data_files = [
      ('share/ament_index/resource_index/packages',
       ['resource/' + package_name]),
      ('share/' + package_name, ['package.xml']),
      (os.path.join('share', package_name), glob("launch/*.launch.py"))
    ],
    install_requires = ['setuptools'],
    zip_safe = True,
    maintainer = 'toby',
    maintainer_email = 't.godfrey@soton.ac.uk',
    description = 'Listener for a Hybrid MARL PoC',
    license = 'TODO: License declaration',
    tests_require = ['pytest'],
    entry_points = {
      'console_scripts': [
        'listener = cmd_listener.listener:main'
      ],
    },
)
