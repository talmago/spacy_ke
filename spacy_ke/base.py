import editdistance

from dataclasses import dataclass
from typing import Dict, Iterator, Tuple, Any, List, Iterable, Set

from spacy import displacy
from spacy.language import Language
from spacy.pipeline.pipes import component
from spacy.tokens.doc import Doc
from spacy.tokens.span import Span
from spacy.util import filter_spans


@dataclass
class Candidate:
    lexical_form: List[str]
    """ the lexical form of the candidate. """

    surface_forms: List[Span]
    """ the surface forms of the candidate. """

    sentence_ids: List[int]
    """ the sentence id of each surface form. """

    @property
    def offsets(self):
        return [sf.start for sf in self.surface_forms]

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
    defaults: Dict[str, Any] = {"candidate_selection": {"ngram": 3}}

    def __init__(self, nlp: Language, **overrides):
        self.nlp = nlp
        self.cfg = self.defaults.copy()
        self.cfg.update(overrides)

    def __call__(self, doc: Doc) -> Doc:
        self.init_component()
        return doc

    def init_component(self):
        if not Doc.has_extension("extract_keywords"):
            Doc.set_extension("extract_keywords", method=self.extract_keywords)
        if not Doc.has_extension("kw_candidates"):
            Doc.set_extension(
                "kw_candidates", setter=self.candidate_selection, getter=self.candidate_selection
            )

    def render(self, doc: Doc, jupyter=None, **kw_kwargs):
        """Render HTML for text highlighting of keywords.

        Args:
            doc (Doc): doc.
            jupyter (bool): optional, override jupyter auto-detection.
            kw_kwargs (kwargs): optional, keyword arguments for ``kw``.

        Returns:
            Rendered HTML markup
        """
        doc = self(doc)
        spans = doc._.extract_keywords(**kw_kwargs)
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
        candidates_weighted = self.candidate_weighting(doc)
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

    def candidate_weighting(self, doc: Doc) -> List[Tuple[Candidate, Any]]:
        """Compute the weighted score of each keyword candidate.

        Args:
            doc (Doc): doc.

        Returns:
            list of tuples, candidate with a score.
        """
        return [(c, 1.0) for c in doc._.kw_candidates]

    def candidate_selection(self, doc: Doc) -> Iterable[Candidate]:
        """Get keywords candidates.

        Args:
            doc (Doc): doc.

        Returns:
            Iterable[Candidate]
        """
        fargs = []
        method = self.cfg["candidate_selection"]
        if hasattr(method, "__call__"):
            f = method
        elif isinstance(method, str):
            f = getattr(self, f"_{method}_selection")
        elif isinstance(method, dict):
            f = getattr(self, f"_{list(method.keys())[0]}_selection")
            args = list(method.values())[0]
            if args is not None:
                if isinstance(args, list):  # sequence of arguments
                    fargs.extend(args)
                else:  # single argument
                    fargs.append(args)
        else:
            raise ValueError(f"Invalid method set in config: {method}")
        return f(doc, *fargs)

    def _chunk_selection(self, doc: Doc) -> Iterable[Candidate]:
        """Get keywords candidates from noun chunks and entities.

        Args:
            doc (Doc): doc.

        Returns:
            Iterable[Candidate]
        """
        surface_forms = []
        spans = list(doc.ents)
        ent_words: Set[str] = set()
        sentence_indices = []
        for span in spans:
            ent_words.update(token.i for token in span)
        for np in doc.noun_chunks:
            # https://github.com/explosion/sense2vec/blob/c22078c4e6c13038ab1c7718849ff97aa54fb9d8/sense2vec/util.py#L105
            while len(np) > 1 and np[0].dep_ not in ("advmod", "amod", "compound"):
                np = np[1:]
            if not any(w.i in ent_words for w in np):
                spans.append(np)
        for sent in doc.sents:
            sentence_indices.append((sent.start, sent.end))
        for span in filter_spans(spans):
            for i, token_indices in enumerate(sentence_indices):
                if span.start >= token_indices[0] and span.end <= token_indices[1]:
                    surface_forms.append((i, span))
                    break
        return self._merge_surface_forms(surface_forms)

    def _ngram_selection(self, doc: Doc, n=3) -> Iterable[Candidate]:
        """Get keywords candidates from ngrams.

        Args:
            doc (Doc): doc.
            n (int): ngram range.

        Returns:
            Iterable[Candidate]
        """
        surface_forms = [sf for sf in self._ngrams(doc, n=n) if self._is_candidate(sf[1])]

        return self._merge_surface_forms(surface_forms)

    @staticmethod
    def _merge_surface_forms(surface_forms: Iterator[Tuple[int, Span]]) -> Iterable[Candidate]:
        """De-dup candidate surface forms.

        Args:
            surface_forms (Iterable): tuples of <sent_i, span>.

        Returns:
            List of candidates.
        """
        candidates = dict()
        for sent_i, span in surface_forms:
            idx = span.lemma_.lower()
            try:
                c = candidates[idx]
            except (KeyError, IndexError):
                lexical_form = [token.lemma_.lower() for token in span]
                c = candidates[idx] = Candidate(lexical_form, [], [])
            c.surface_forms.append(span)
            c.sentence_ids.append(sent_i)
        return list(candidates.values())

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
