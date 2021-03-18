with import <nixpkgs> {};

let
  pythonEnv = python37;
  pythonPackages = python37Packages;

in mkShell {
  buildInputs = [
    pythonEnv
    pythonPackages.pip-tools
    pythonPackages.setuptools
    pythonPackages.black
    pythonPackages.ipython
    pythonPackages.pytest
    pipenv
    which
    gcc
    binutils
  ];

  shellHook = ''
     # set $PYTHONPATH
     export PYTHONPATH=$PYTHONPATH:$(pwd);

     # install python dependencies
     pipenv sync -d;

     # install spaCy model
     pipenv run python -m spacy download en_core_web_sm
  '';
}
