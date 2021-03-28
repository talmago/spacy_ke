import spacy

import spacy_ke

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("yake")

doc = nlp(
    "Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence "
    "concerned with the interactions between computers and human language, in particular how to program computers "
    "to process and analyze large amounts of natural language data. "
)

for keyword, score in doc._.extract_keywords(n=3):
    print(keyword, "-", score)
