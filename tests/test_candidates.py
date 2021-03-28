from spacy_ke.util import chunk_selection, ngram_selection


def test_ngram_range_candidate_selection(en_core_web_sm):
    doc = en_core_web_sm(
        "Natural language processing is fun"
    )

    candidates = ngram_selection(doc, n=3)
    assert candidates[0].lexical_form == ["natural"]
    assert candidates[0].sentence_ids == [0]
    assert candidates[0].surface_forms[0].start == 0
    assert candidates[0].surface_forms[0].end == 1


def test_chunk_selection(en_core_web_sm):
    doc = en_core_web_sm(
        "Natural language processing is fun"
    )

    candidates = chunk_selection(doc)
    assert candidates[0].lexical_form == ["natural", "language", "processing"]
    assert candidates[0].sentence_ids == [0]
    assert candidates[0].surface_forms[0].start == 0
    assert candidates[0].surface_forms[0].end == 3
