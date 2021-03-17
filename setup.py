#!/usr/bin/env python
# -*- coding: utf-8 -*-
# inspired from:
# https://github.com/kennethreitz/setup.py/blob/master/setup.py
# Note: To use the 'upload' functionality of this file, you must:
#   $ pipenv install twine --dev

import io
import os
import sys
from shutil import rmtree
from setuptools import find_packages, setup, Command

root = os.path.abspath(os.path.dirname(__file__))
sys.path.append(root)

with io.open(os.path.join(root, "spacy_ke", "about.py"), encoding="utf8") as f:
    about = {}
    exec(f.read(), about)

# Import the README and use it as the long-description.
with io.open(os.path.join(root, "README.md"), encoding="utf-8") as f:
    long_description = "\n" + f.read()


class UploadCommand(Command):
    """Support setup.py upload."""

    description = "Build and publish the package."
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print("\033[1m{0}\033[0m".format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status("Removing previous builds…")
            rmtree(os.path.join(root, "dist"))
        except OSError:
            pass

        self.status("Building Source and Wheel distribution…")
        os.system("{0} setup.py sdist bdist_wheel".format(sys.executable))

        self.status("Uploading the package to PyPI via Twine…")
        os.system("twine upload dist/*")
        sys.exit()


setup(
    name=about["__title__"],
    version=about["__version__"],
    description=about["__summary__"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=about["__author__"],
    author_email=about["__email__"],
    url=about["__url__"],
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    install_requires=["editdistance", "numpy", "networkx", "spacy<2.3.0", "scipy"],
    python_requires=">=3.7",
    include_package_data=True,
    license=about["__license__"],
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering",
    ],
    # $ setup.py publish support.
    cmdclass={
        "upload": UploadCommand,
    },
    entry_points={}
)
