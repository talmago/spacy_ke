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


if __name__ == "__main__":

    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("yake")

    doc = nlp(
        "Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence "
        "concerned with the interactions between computers and human language, in particular how to program computers "
        "to process and analyze large amounts of natural language data. "
    )

    for keyword, score in doc._.extract_keywords(n=3):
        print(keyword, "-", score)
