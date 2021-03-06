import spacy
import spacy_ke
import pdb

try:
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("ngram_kw_candidates")
    yake = nlp.add_pipe("yake_keyword_extractor")
    doc = nlp(
        """
        Natural language processing (NLP) is a subfield of linguistics,
        computer science, and artificial intelligence concerned with the
        interactions between computers and human language, in particular how
        to program computers to process and analyze large amounts of natural
        language data.
        """
    )
    print(doc._.extract_keywords())
except Exception as exn:
    print(str(exn).strip())
    pdb.post_mortem()