import pdb

import spacy

from spacy_ke import KWCandidates

try:
    nlp = spacy.load("en_core_web_sm")
    # add with a keyword candidate filter specified as configuration for the
    # extractor. NOTE: this does not mutate the language pipeline until it is
    # called for the first time, it's only implemented for backward
    # compatibility, and you probably shouldn't do it
    yake = nlp.add_pipe(
        "yake_keyword_extractor", config={"candidate_selector": KWCandidates.NOUN_CHUNKS.value}
    )
    # ...instead of explicitly specifying "config", in new code, you can just
    # as easily (and more succinctly)...
    nlp.add_pipe(KWCandidates.NGRAM.value, before="yake_keyword_extractor")
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
