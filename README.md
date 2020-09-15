# spacy_yake: Yake! Keyword Extraction for spaCy.

Inspired by [pke](https://github.com/boudinfl/pke).

## ⏳ Installation

```bash
pip install spacy_yake
```

## 🚀 Quickstart

### Usage as a spaCy pipeline component

```python
import spacy

from spacy_yake import Yake

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe(Yake(nlp))

doc = nlp(
    "Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence "
    "concerned with the interactions between computers and human language, in particular how to program computers "
    "to process and analyze large amounts of natural language data. "
)

for keyword, score in doc._.kw(n=3):
    print(keyword, "-", score)

# Output:
# computer science - 0.020279855002262884
# NLP - 0.035016746977200745
# Natural language processing - 0.04407186487965091
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
$ pipenv run black spacy_yake/ --config=pyproject.toml
```

## References

[1] YAKE! Collection-Independent Automatic Keyword Extractor

```
@InProceedings{10.1007/978-3-319-76941-7_80,
author="Campos, Ricardo
and Mangaravite, V{\'i}tor
and Pasquali, Arian
and Jorge, Al{\'i}pio M{\'a}rio
and Nunes, C{\'e}lia
and Jatowt, Adam",
editor="Pasi, Gabriella
and Piwowarski, Benjamin
and Azzopardi, Leif
and Hanbury, Allan",
title="YAKE! Collection-Independent Automatic Keyword Extractor",
booktitle="Advances in Information Retrieval",
year="2018",
publisher="Springer International Publishing",
address="Cham",
pages="806--810",
abstract="In this paper, we present YAKE!, a novel feature-based system for multi-lingual keyword extraction from single documents, which supports texts of different sizes, domains or languages. Unlike most systems, YAKE! does not rely on dictionaries or thesauri, neither it is trained against any corpora. Instead, we follow an unsupervised approach which builds upon features extracted from the text, making it thus applicable to documents written in many different languages without the need for external knowledge. This can be beneficial for a large number of tasks and a plethora of situations where the access to training corpora is either limited or restricted. In this demo, we offer an easy to use, interactive session, where users from both academia and industry can try our system, either by using a sample document or by introducing their own text. As an add-on, we compare our extracted keywords against the output produced by the IBM Natural Language Understanding (IBM NLU) and Rake system. YAKE! demo is available at http://bit.ly/YakeDemoECIR2018. A python implementation of YAKE! is also available at PyPi repository (https://pypi.python.org/pypi/yake/).",
isbn="978-3-319-76941-7"
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

[3] A Review of Keyphrase Extraction

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