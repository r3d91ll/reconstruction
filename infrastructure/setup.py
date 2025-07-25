"""
Setup script for Information Reconstructionism Infrastructure
"""

from setuptools import setup, find_packages

setup(
    name="irec-infrastructure",
    version="0.1.0",
    description="Infrastructure components for Information Reconstructionism experiments",
    author="Information Reconstructionism Project",
    packages=find_packages(include=["irec_infrastructure", "irec_infrastructure.*"]),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
        "python-arango>=7.5.0",
        "httpx>=0.24.0",
        "docling>=1.0.0",
        "jina>=3.20.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "mypy>=1.0.0",
        ]
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)