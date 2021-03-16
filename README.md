# spacy_ke: Keyword Extraction with spaCy.

## ⏳ Installation

```bash
pip install spacy_ke
```

## 🚀 Quickstart

### Usage as a spaCy pipeline component (spaCy v2.x.x)

```python
import spacy

from spacy_ke import Yake

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe(Yake(nlp))

doc = nlp(
    "Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence "
    "concerned with the interactions between computers and human language, in particular how to program computers "
    "to process and analyze large amounts of natural language data. "
)

for keyword, score in doc._.extract_keywords(n=3):
    print(keyword, "-", score)

# computer science - 0.020279855002262884
# NLP - 0.035016746977200745
# Natural language processing - 0.04407186487965091
```

### Usage as a spaCy pipeline component (spaCy v3.x.x)

```python
import spacy

from spacy.language import Language
from spacy_ke import Yake


@Language.factory(
    "yake", default_config={"window": 2, "lemmatize": False, "candidate_selection": "ngram"}
)
def yake(nlp, name, window: int, lemmatize: bool, candidate_selection: str):
    return Yake(
        nlp, window=window, lemmatize=lemmatize, candidate_selection=candidate_selection
    )

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("yake")
```


### Configure the pipeline component

Normally you'd want to configure the keyword extraction pipeline according to its implementation. 

```python
window: int = 2 # default
lemmatize: bool = False  # default
candidate_selection: str = "ngram" # default, use "chunk" for noun phrase selection.

nlp.add_pipe(
    Yake(
        nlp, 
        window=window, # default
        lemmatize=lemmatize, # default
        candidate_selection="ngram" # default, use "chunk" for noun phrase selection
    )
)
```

And if you want to define a custom candidate selection use the example below.

```python
from spacy_ke.util import registry, Candidate

@registry.candidate_selection.register("custom")
def custom_selection(doc: Doc, n=3) -> Iterable[Candidate]:
  ...

nlp.add_pipe(
    Yake(
        nlp, 
        candidate_selection="custom"
    )
)

```

## Development

Set up pip & virtualenv

```sh
$ pipenv sync -d
```

Run unit test

```sh
$ pipenv run pytest
```

Run black (code formatter)

```sh
$ pipenv run black spacy_ke/ --config=pyproject.toml
```

Release package (via `twine`)

```sh
$ python setup.py upload
```

## References

[1] A Review of Keyphrase Extraction

```
@article{DBLP:journals/corr/abs-1905-05044,
  author    = {Eirini Papagiannopoulou and
               Grigorios Tsoumakas},
  title     = {A Review of Keyphrase Extraction},
  journal   = {CoRR},
  volume    = {abs/1905.05044},
  year      = {2019},
  url       = {http://arxiv.org/abs/1905.05044},
  archivePrefix = {arXiv},
  eprint    = {1905.05044},
  timestamp = {Tue, 28 May 2019 12:48:08 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-1905-05044.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

[2] [pke](https://github.com/boudinfl/pke): an open source python-based keyphrase extraction toolkit.

```
@InProceedings{boudin:2016:COLINGDEMO,
  author    = {Boudin, Florian},
  title     = {pke: an open source python-based keyphrase extraction toolkit},
  booktitle = {Proceedings of COLING 2016, the 26th International Conference on Computational Linguistics: System Demonstrations},
  month     = {December},
  year      = {2016},
  address   = {Osaka, Japan},
  pages     = {69--73},
  url       = {http://aclweb.org/anthology/C16-2015}
}
```