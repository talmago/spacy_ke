import networkx as nx
import numpy as np

from typing import Dict, Tuple, Any, List
from spacy.tokens.doc import Doc

from spacy_ke.base import KeywordExtractor
from spacy_ke.util import Candidate


class TextRank(KeywordExtractor):
    """spaCy implementation of "TextRank: Bringing Order into Text".

    Usage example:
    --------------

    >>> import spacy

    >>> nlp = spacy.load("en_core_web_sm")
    >>> nlp.add_pipe(TextRank(nlp))

    >>> doc = nlp(
        "Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence "
        "concerned with the interactions between computers and human language, in particular how to program computers "
        "to process and analyze large amounts of natural language data. "
    )

    >>> doc._.extract_keywords(n=5)
    """

    defaults: Dict[str, Any] = {
        "pos": frozenset({"ADJ", "NOUN", "PROPN", "VERB"}),
        "window": 3,
        "alpha": 0.85,
        "tol": 1.0e-6,
        "candidate_selection": "chunk",
    }

    def candidate_weighting(self, doc: Doc) -> List[Tuple[Candidate, float]]:
        """Compute the weighted score of each keyword candidate.

        Args:
            doc (Doc): doc.

        Returns:
            list of tuples, candidate with a score.
        """
        res = []
        G = self.build_graph(doc)
        W = nx.pagerank_scipy(G, alpha=self.cfg["alpha"], tol=self.cfg["tol"])
        for candidate in doc._.kw_candidates:
            chunk_len = len(candidate.lexical_form)
            non_lemma = 0
            rank = 0.0
            for t in candidate.lexical_form:
                if t in W:
                    rank += W[t]
                else:
                    non_lemma += 1
            non_lemma_discount = chunk_len / (chunk_len + (2.0 * non_lemma) + 1.0)
            candidate_w = np.sqrt(rank / (chunk_len + non_lemma)) * non_lemma_discount
            candidate_w += candidate.offsets[0] * 1e-8  # break ties according to position in text
            res.append((candidate, candidate_w))
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
        pos = self.cfg["pos"]
        window_size = self.cfg["window"]
        seen = set()
        for sent in doc.sents:
            for token in sent:
                if token.is_stop or token.pos_ not in pos:
                    continue
                node0 = token.lemma_.lower()
                if not G.has_node(node0):
                    G.add_node(node0)
                for prev_token in sent[max(sent.start, token.i - window_size) : token.i]:
                    node1 = prev_token.lemma_.lower()
                    if node0 != node1 and node1 in seen:
                        if G.has_edge(node0, node1):
                            G[node0][node1]["weight"] += 1
                        else:
                            G.add_edge(node0, node1, weight=1)
                seen.add(node0)
        return G
