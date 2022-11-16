from setuptools import setup, find_packages

VERSION = '0.0.7'
DESCRIPTION = 'A package for identify Machine Learning bugs'

# Setting up
setup(
    name="mlbugdetection",
    version=VERSION,
    author="João Gianfaldoni | Giovanni Cardoso | William Silva",
    author_email="william.silva.ismart@gmail.com",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=["pandas", "numpy", "matplotlib", "sklearn"],
    keywords=['python'],
    include_package_data=True,
    package_data={'': ['data/*.csv', 'models/*.pkl']},
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)