"""
SEAM-AVDA: Esoteric language for non-orientable computing + Antipodal Vector Database
"""
from setuptools import setup, find_packages, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
import sys

# C++ extension modules
ext_modules = [
    Pybind11Extension(
        "libantipodal_core",
        ["src/libantipodal/quotient_space.cpp",
         "src/libantipodal/parity_index.cpp",
         "src/libantipodal/backends/faiss_backend.cpp"],
        include_dirs=["src/libantipodal"],
        extra_compile_args=["-std=c++17", "-O3"],
        define_macros=[("VERSION_INFO", '"0.1.0"')],
    ),
]

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="seam-avda",
    version="0.1.0",
    author="SEAM-AVDA Contributors",
    description="Esoteric language for non-orientable computing + Antipodal Vector Database",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MacMayo1993/SEAM-AVDA",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: C++",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Database",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pybind11>=2.10.0",
        "faiss-cpu>=1.7.4",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "mypy>=0.950",
        ],
        "examples": [
            "sentence-transformers>=2.2.0",
            "scikit-learn>=1.0.0",
            "matplotlib>=3.5.0",
        ],
    },
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
