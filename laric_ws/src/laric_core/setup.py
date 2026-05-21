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
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='lourdes',
    maintainer_email='lourdesfll29@gmail.com',
    description='LARIC: Language-based Agent for Robotic Interaction and Control',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            # Intentionally left blank for Bash script execution
        ],
    },
)
