import os
from setuptools import setup, find_packages


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

README = open(os.path.join(SCRIPT_DIR, "README.md")).read()


setup_dict = dict(
    name="example-gridworld",
    version="0.1.0",
    description="Example gridworld reinforcement learning environment",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/christopher-hesse/example-gridworld",
    author="Christopher Hesse",
    license="Public Domain",
    packages=find_packages(),
    install_requires=[
        "gym3>=0.3.1,<0.4.0",
        "numpy>=1.18.0,<2.0.0",
        "imageio>=2.8.0,<3.0.0",
    ],
    extras_require={"dev": ["pytest", "pytest-benchmark"]},
    python_requires=">=3.7.0",
)

setup(**setup_dict)
