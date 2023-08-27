"""Setup module."""
from setuptools import setup, find_packages

VERSION = '0.0.12'
DESCRIPTION = 'A package for identify Machine Learning bugs'

with open('README.md', 'r', encoding="utf-8") as f:
    LONG_DESCRIPTION = f.read()

with open('requirements.txt', 'r', encoding="utf-8") as f:
    INSTALL_REQUIRES = f.read().strip().split('\n')

# Setting up
setup(
    name="mlbugdetection",
    version=VERSION,
    author="João Gianfaldoni | Giovanni Cardoso | William Silva",
    author_email="william.silva.ismart@gmail.com",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    keywords=['python', 'machine learning', 'bug detection'],
    include_package_data=True,
    package_data={'': ['data/*.csv', 'models/*.pkl']},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)