from abc import ABC
from dataclasses import dataclass
from collections import deque
from spacy.language import Language
from spacy.tokens.doc import Doc
from spacy.tokens.span import Span
from spacy.util import filter_spans
from typing import Any, Iterable, Iterator, Tuple, List, Set
import editdistance


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


class KeywordCandidates(ABC):
    def __init__(self, nlp: Language, name: str):
        self.nlp = nlp
        self.component_name = name

    def __call__(self, doc: Doc) -> Doc:
        if not hasattr(doc._, "kw_candidates"):
            Doc.set_extension("kw_candidates", getter=self.extract_candidates)
        doc._.kw_candidates = self.extract_candidates(doc)
        return doc

    def __init_subclass__(SelectorImpl):
        selector_name = SelectorImpl.__name__
        selector_name = selector_name[: -len("KeywordCandidates")]
        selector_name = "".join(c if c.islower() else "_" + c.lower() for c in selector_name)
        if selector_name.startswith("_"):
            selector_name = selector_name[1:]
        component_name = f"{selector_name}_kw_candidates"
        Language.factory(
            name=component_name,
            requires=["token.pos", "token.dep"],
            assigns=["doc._.kw_candidates"],
            func=SelectorImpl,
        )

    def extract_candidates(self, doc: Doc) -> Iterator[Candidate]:
        pass

    @staticmethod
    def merge_surface_forms(surface_forms: Iterator[Tuple[int, Span]]) -> Iterable[Candidate]:
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


class NounChunkKeywordCandidates(KeywordCandidates):
    """Get keywords candidates from noun chunks and entities."""

    def extract_candidates(self, doc):
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
        return surface_forms


class NgramKeywordCandidates(KeywordCandidates):
    def __init__(
        self,
        nlp: Language,
        name: str,
        min_candidate_tokens=3,
        min_token_length=2,
        alpha_tokens_only=True,
    ):
        self.min_candidate_tokens = 3
        self.min_token_length = 2
        self.alpha_tokens_only = True

    def ngrams(self, doc: Doc) -> Iterator[Tuple[int, Span]]:
        """Select all the n-grams and populate the candidate container.

        Args:
            doc (Doc): doc.
            n (int): the n-gram length, defaults to 3.

        Returns:
            Iterator(sentence_id<int>, offset<int>, ngram<Span>)
        """
        for sentence_id, sentence in enumerate(doc.sents):
            n_tokens = len(sentence)
            window = min(self.min_candidate_tokens, n_tokens)
            for j in range(n_tokens):
                for k in range(j + 1, min(j + 1 + window, n_tokens + 1)):
                    yield sentence_id, sentence[j:k]

    def is_candidate(self, span: Span) -> bool:
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
        if len(span.text) < self.min_candidate_tokens:
            return False
        for token_idx, token in enumerate(span):
            # discard if contains punct
            if token.is_punct:
                return False
            # discard if contains tokens of 1-2 characters
            if len(token.text) < self.min_token_length:
                return False
            # discard if contains non alphanumeric
            if self.alpha_tokens_only and not token.is_alpha:
                return False
            # discard if first/last word is a stop
            if token_idx in (0, n_span - 1) and token.is_stop:
                return False
        # discard if ends with NOUN -> VERB
        if n_span >= 2 and span[-2].pos_ == "NOUN" and span[-1].pos_ == "VERB":
            return False
        return True

    def extract_candidates(self, doc: Doc) -> Iterator[Candidate]:
        surface_forms = [sf for sf in self.ngrams(doc) if self.is_candidate(sf[1])]
        return self.merge_surface_forms(surface_forms)