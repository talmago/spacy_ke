import editdistance

from dataclasses import dataclass
from typing import Dict, Iterator, Tuple, Any, List, Iterable, Set
from abc import ABC

from inspect import signature, Parameter
from functools import wraps

from spacy import displacy
from spacy.language import Language
from spacy.tokens.doc import Doc
from spacy.tokens.span import Span

from .candidates import Candidate


class KeywordExtractor(ABC):
    def __init__(self, nlp: Language, name: str):
        self.nlp = nlp
        self.component_name = name

    def __call__(self, doc: Doc) -> Doc:
        self.init_component()
        print(doc)
        return doc

    def __init_subclass__(ExtractorImplementation):
        """
        Decorate each keyword extractor implementation as a spaCy pipeline
        component with a name corresponding to class-name of the extractor,
        e.g., positionrank_keyword_extractor for a subclass named PositionRank.
        """
        extractor_kind = ExtractorImplementation.__name__.lower()
        component_name = f"{extractor_kind}_keyword_extractor"

        Language.factory(
            component_name,
            requires=["token.pos", "token.dep", "doc.sents", "doc._.kw_candidates"],
            func=ExtractorImplementation,
        )

    def init_component(self):
        if not Doc.has_extension("extract_keywords"):
            Doc.set_extension("extract_keywords", method=self.extract_keywords)

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