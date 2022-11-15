from distutils.core import setup, Extension
from setuptools import find_packages
import pybind11
import sys


dependencies = ['tensorflow', 'pybind11']

cpp_args = ['-std=c++11', '-stdlib=libc++', '-mmacosx-version-min=10.7']

sfc_module = Extension('flame',
                        sources=['./env/TrafficSim.cpp'],
                        language='c++',
                        include_dirs=[pybind11.get_include()],
                        extra_compile_args=cpp_args
                        )

setup(
    name = 'flame',
    version='1.0.0',
    packages=find_packages(),
    description='traffic light control using rl',
    long_description="Traffic Light Control in Traffic Simulation using MADDPG",
    install_requires=dependencies,
    ext_modules=[sfc_module]
)

