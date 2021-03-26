with import <nixpkgs> {};

let
  pythonEnv = python37;
  pythonPackages = python37Packages;

in mkShell {
  buildInputs = [
    pythonEnv
    pythonPackages.pip-tools
    pythonPackages.setuptools
    which
    gcc
    binutils
  ];

  shellHook = ''
     # create virtual env
     python -m venv .venv

     # activate virtualenv
     source .venv/bin/activate

     # install python dependencies
     pip install -U pip && pip install -r requirements-dev.txt

     # install spaCy model
     python -m spacy download en_core_web_sm

     # set $PYTHONPATH
     export PYTHONPATH=$PYTHONPATH:$(pwd);
  '';
}
