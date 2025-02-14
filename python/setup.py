#
#           Python Library Interface
#           Recurrence Microstates - Python
#           Created by Gabriel Ferreira on February 2025.
#           Advisors: Thiago de Lima Prado and Sérgio Roberto Lopes
#           Federal University of Paraná - Physics Department
#
#       Julia version: https://github.com/gabriel-ferr/RecurrenceMicrostates.jl
#       C++ version: https://github.com/gabriel-ferr/RecurrenceMicrostates
#
#       --- Need libraries...
from pydoc import describe

from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        "recurrms",
        ["lib/lib.cpp"],
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=["-std=c++23"],
    ),
]

setup(
    name="recurrms",
    version="0.0.1",
    author="Gabriel Ferreira",
    description="A library to compute the recurrence microstates probabilities from some generic data.",
    ext_modules=ext_modules,
    zip_safe=False
)