import os, io

from setuptools import find_packages, setup

def read(rel_path: str) -> str:
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, rel_path)) as fp:
        return fp.read()


def get_version(rel_path: str) -> str:
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")


version = get_version("ddl/__about__.py")

# with io.open("README.md", "r", encoding="utf-8") as f:
#     long_description = f.read()

# with open("requirements.txt", "r") as f:
#     install_requires = [x.strip() for x in f.readlines()]

setup(
    name="DistributedDataLoader",
    version=version,
    description="An MPI shared memory implementation of a PyTorch DataLoader",
    # long_description=long_description,
    long_description_content_type="text/markdown",
    license="LGPL-2.1",
    # install_requires=install_requires,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU Lesser General Public License v2 (LGPLv2)",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords=[
        "Scientific machine learning",
        "Machine learning",
        "Deep learning",
        "Neural networks",
    ],
    packages=find_packages(),
)
