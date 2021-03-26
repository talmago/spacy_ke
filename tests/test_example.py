from spacy_ke import Yake


def test_example(en_core_web_sm, spacy_v3):
    nlp = en_core_web_sm

    if spacy_v3:
        from spacy.language import Language

        @Language.factory(
            "yake", default_config={
                "window": 2,
                "lemmatize": False,
                "candidate_selection": "ngram"
            }
        )
        def yake(nlp, name, window: int, lemmatize: bool, candidate_selection: str):
            return Yake(
                nlp,
                window=window,
                lemmatize=lemmatize,
                candidate_selection=candidate_selection
            )

        nlp.add_pipe("yake")


    else:
        nlp.add_pipe(Yake(nlp))

    doc = nlp(
        "Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence "
        "concerned with the interactions between computers and human language, in particular how to program computers "
        "to process and analyze large amounts of natural language data. "
    )

    keywords_with_scores = doc._.extract_keywords(n=3)

    assert keywords_with_scores[0][0].text == "computer science"
    assert keywords_with_scores[1][0].text == "NLP"
    assert keywords_with_scores[2][0].text == "Natural language processing"
