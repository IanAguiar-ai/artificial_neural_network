from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = '0.0.1'
DESCRIPTION = 'Artificial Neural Network'
LONG_DESCRIPTION = 'Artificial neural network made from scratch'

# Setting up
setup(
    name="artificial_neural_network",
    version=VERSION,
    author="Ian dos Anjos",
    author_email="<iannaianjos@gmail.com>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    install_requires=[''],
    keywords=['python', 'artificial_neural_network', 'neural_network', 'mlp', 'multilayer_perceptron'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)
