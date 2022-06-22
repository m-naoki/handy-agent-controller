from setuptools import setup, find_packages


install_requires = [
    'opencv-python',
    'mediapipe',
    'hydra'
]


setup(
    name='handy_rl_controller',
    version='0.0.0',
    description="controller for RL capturing hand pose by web-camera",
    url="",
    license="MIT",
    packages=find_packages(),
    install_requires=install_requires,
    entry_points={},
)