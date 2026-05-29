# LARIC System - Package Setup
# Copyright (c) 2026 LARIC. All rights reserved.
#
# Configuration for the laric_core ROS 2 package. Defines package metadata,
# dependencies, and executable entry points for the colcon build system.

from setuptools import find_packages, setup

package_name = 'laric_core'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    	('share/' + package_name + '/config', ['config/laric_nav2_params.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='lourdes',
    maintainer_email='lourdesfll29@gmail.com',
    description='LARIC: Language-based Agent for Robotic Interaction and ' \
                'Control',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            # Intentionally left blank for Bash script execution
        ],
    },
)
