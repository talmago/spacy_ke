from functools import total_ordering
from typing import Any, Dict, Iterable, List, Tuple

import networkx as nx
from spacy.language import Language
from spacy.tokens.doc import Doc

from spacy_ke.base import Candidate, KeywordExtractor, KWCandidates


class PositionRank(KeywordExtractor):
    """spaCy implementation of "PositionRank: An Unsupervised Approach to Keyphrase Extraction from Scholarly".

    Usage example:
    --------------

    >>> import spacy

    >>> nlp = spacy.load("en_core_web_sm")
    >>> nlp.add_pipe(PositionRank(nlp))

    >>> doc = nlp(
        "Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence "
        "concerned with the interactions between computers and human language, in particular how to program computers "
        "to process and analyze large amounts of natural language data. "
    )

    >>> doc._.extract_keywords(n=5)
    """

    def __init__(
        self,
        nlp: Language,
        name: str,
        candidate_selector: KWCandidates = KWCandidates.noun_chunks,
        permitted_pos: Iterable[str] = frozenset({"ADJ", "NOUN", "PROPN"}),
        window=10,
        alpha=0.85,
        tol=1.0e-5,
        normalize=False,
    ):
        super().__init__(nlp, name, candidate_selector)
        self.nlp = nlp
        self.component_name = name
        self.candidate_selector = candidate_selector
        self.pos = permitted_pos
        self.window = alpha
        self.alpha = alpha
        self.tol = tol
        self.normalize = normalize

    def candidate_weighting(self, doc: Doc) -> List[Tuple[Candidate, float]]:
        """Compute the weighted score of each keyword candidate.

        Args:
            doc (Doc): doc.

        Returns:
            list of tuples, candidate with a score.
        """
        res = []

        # build the word graph
        G = self.build_graph(doc)

        # normalize cumulated inverse positions
        positions = nx.get_node_attributes(G, "position")
        norm = sum(positions.values())
        for word in positions:
            positions[word] /= norm

        # compute the word scores using biased random walk
        W = nx.pagerank_scipy(
            G, alpha=self.alpha, tol=self.tol, personalization=positions, weight="weight"
        )

        for candidate in doc._.kw_candidates:
            candidate_w = sum([W.get(t, 0.0) for t in candidate.lexical_form])
            if self.normalize:
                candidate_w /= len(candidate.lexical_form)
            res.append((candidate, candidate_w))

        # sort by score
        res.sort(key=lambda x: x[1], reverse=True)
        return res

    def build_graph(self, doc: Doc):
        """Build a lemma graph representation of a document.

        Args:
            doc (Doc): doc.

        Returns:
            nx.Graph
        """
        G = nx.Graph()
        pos = self.pos
        window_size = self.window
        n_tokens = len(doc)

        for token in doc:
            if token.pos_ not in pos:
                continue
            node0 = token.lemma_.lower()
            position0 = token.i
            if not G.has_node(node0):
                G.add_node(node0, position=1 / (position0 + 1))

            j = position0 + 1
            while j < n_tokens and (doc[j].i - token.i) < window_size:
                node1, position1 = doc[j].lemma_.lower(), doc[j].i
                if node0 != node1:
                    if not G.has_node(node1):
                        G.add_node(node1, position=(1 / position1 + 1))
                    if not G.has_edge(node0, node1):
                        G.add_edge(node0, node1, weight=0)
                    G[node0][node1]["weight"] += 1
                j = j + 1

        return G
