import editdistance

from dataclasses import dataclass
from typing import Dict, Iterator, Tuple, Any, List, Iterable

from spacy import displacy
from spacy.language import Language
from spacy.pipeline.pipes import component
from spacy.tokens.doc import Doc
from spacy.tokens.span import Span


@dataclass
class Candidate:
    lexical_form: List[str]
    """ the lexical form of the candidate. """

    surface_forms: List[Span]
    """ the surface forms of the candidate. """

    offsets: List[List[int]]
    """ the offsets of the surface forms. """

    sentence_ids: List[int]
    """ the sentence id of each surface form. """

    def similarity(self, other):
        """Measure a similarity to another candidate.

        ``SIM = 1 - edit_distance(lexical_form1, lexical_form2) / max(#lexical_form1, #lexical_form2)``.
        """
        assert isinstance(other, Candidate)

        lf1 = " ".join(self.lexical_form)
        lf2 = " ".join(other.lexical_form)
        dist = editdistance.eval(lf1, lf2)
        dist /= max(len(lf1), len(lf2))
        return 1.0 - dist


@component(
    "keyword_extractor",
    requires=["token.pos", "token.dep", "doc.sents"],
    assigns=["doc._.extract"],
)
class KeywordExtractor:
    cfg: Dict[str, Any] = {"ngram": 3}

    def __init__(self, nlp: Language, **overrides):
        self.nlp = nlp
        self.cfg.update(overrides)

    def __call__(self, doc: Doc) -> Doc:
        self.init_component()
        return doc

    def init_component(self):
        if not Doc.has_extension("extract_keywords"):
            Doc.set_extension("extract_keywords", method=self.extract_keywords)
        if not Doc.has_extension("kw_candidates"):
            Doc.set_extension("kw_candidates", getter=self.get_candidates)

    def render(self, doc: Doc, jupyter=None, **kw_kwargs):
        """Render HTML for text highlighting of keywords.

        Args:
            doc (Doc): doc.
            jupyter (bool): optional, override jupyter auto-detection.
            kw_kwargs (kwargs): optional, keyword arguments for ``kw``.

        Returns:
            Rendered HTML markup
        """
        spans = self(doc)._.extract_keywords(**kw_kwargs)
        examples = [
            {
                "text": doc.text,
                "title": None,
                "ents": sorted(
                    [
                        {
                            "start": span.start_char,
                            "end": span.end_char,
                            "label": f"{round(score, 3)}",
                        }
                        for (span, score) in spans
                    ],
                    key=lambda x: x["start"],
                ),
            }
        ]
        html = displacy.render(examples, style="ent", manual=True, jupyter=jupyter)
        return html

    def extract_keywords(self, doc: Doc, n=10, similarity_thresh=0.45):
        """Returns the n-best candidates given the weights.

        Args:
            doc (Doc): doc.
            n (int): the number of candidates, defaults to 10.
            similarity_thresh (float): optional, similarity thresh for dedup. default is 0.45.

        Returns:
            List
        """
        spans = []
        candidates_weighted = self.weight_candidates(doc)
        candidates_weighted.sort(key=lambda x: x[1])
        for candidate, candidate_w in candidates_weighted:
            if similarity_thresh > 0.0:
                redundant = False
                for prev_candidate, _ in spans:
                    if candidate.similarity(prev_candidate) > similarity_thresh:
                        redundant = True
                        break
                if redundant:
                    continue
            spans.append((candidate, candidate_w))
            if len(spans) >= n:
                break
        spans = spans[: min(n, len(spans))]
        spans = [(c.surface_forms[0], score) for c, score in spans]
        return spans

    def weight_candidates(self, doc: Doc) -> List[Tuple[Candidate, Any]]:
        """Compute the weighted score of each keyword candidate.

        Args:
            doc (Doc): doc.

        Returns:
            list of tuples, candidate with a score.
        """
        return [(c, 1.0) for c in doc._.kw_candidates]

    def get_candidates(self, doc: Doc) -> Iterable[Candidate]:
        """Get keywords candidates.

        Args:
            doc (Doc): doc.

        Returns:
            Iterable[Candidate]
        """
        candidates = dict()
        for sentence_id, ngram_span in self._ngrams(doc, n=self.cfg["ngram"]):
            if not self._is_candidate(ngram_span):
                continue
            idx = ngram_span.lemma_
            try:
                c = candidates[idx]
            except (KeyError, IndexError):
                lexical_form = [token.lemma_.lower() for token in ngram_span]
                c = candidates[idx] = Candidate(lexical_form, [], [], [])
            c.surface_forms.append(ngram_span)
            c.sentence_ids.append(sentence_id)
        return candidates.values()

    @staticmethod
    def _ngrams(doc: Doc, n=3) -> Iterator[Tuple[int, Span]]:
        """Select all the n-grams and populate the candidate container.

        Args:
            doc (Doc): doc.
            n (int): the n-gram length, defaults to 3.

        Returns:
            Iterator(sentence_id<int>, offset<int>, ngram<Span>)
        """
        for sentence_id, sentence in enumerate(doc.sents):
            n_tokens = len(sentence)
            window = min(n, n_tokens)
            for j in range(n_tokens):
                for k in range(j + 1, min(j + 1 + window, n_tokens + 1)):
                    yield sentence_id, sentence[j:k]

    @staticmethod
    def _is_candidate(span: Span, min_length=3, min_word_length=2, alpha=True) -> bool:
        """Check if N-gram span is qualified as a candidate.

        Args:
            span (Span): n-gram.
            min_length (int): minimum length for n-gram.
            min_word_length (int): minimum length for word in an ngram.
            alpha (bool): Filter n-grams with non-alphanumeric words.

        Returns:
            bool
        """
        n_span = len(span)
        # discard if composed of 1-2 characters
        if len(span.text) < min_length:
            return False
        for token_idx, token in enumerate(span):
            # discard if contains punct
            if token.is_punct:
                return False
            # discard if contains tokens of 1-2 characters
            if len(token.text) < min_word_length:
                return False
            # discard if contains non alphanumeric
            if alpha and not token.is_alpha:
                return False
            # discard if first/last word is a stop
            if token_idx in (0, n_span - 1) and token.is_stop:
                return False
        # discard if ends with NOUN -> VERB
        if n_span >= 2 and span[-2].pos_ == "NOUN" and span[-1].pos_ == "VERB":
            return False
        return True
