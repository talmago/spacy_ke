with import <nixpkgs> {};

let
  pythonEnv = python37;

in mkShell {
  buildInputs = [
    python37
    python37Packages.pip-tools
    python37Packages.setuptools
    python37Packages.black
    python37Packages.ipython
    python37Packages.pytest

    pipenv
    which
    gcc
    binutils
  ];
}
