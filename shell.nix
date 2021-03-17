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

  shellHook = ''
     # set $PYTHONPATH
     export PYTHONPATH=$PYTHONPATH:$(pwd);

     # install python dependencies
     pipenv sync -d;

     # install spaCy model
     pipenv run python -m spacy download en_core_web_sm
  '';
}
