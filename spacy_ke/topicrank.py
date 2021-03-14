import itertools
import networkx as nx
import numpy as np

from typing import Dict, Tuple, Any, List, Iterable

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from spacy.tokens.doc import Doc

from spacy_ke.base import KeywordExtractor
from spacy_ke.util import Candidate


class TopicRank(KeywordExtractor):
    """spaCy implementation of "TopicRank: Graph-Based Topic Ranking for Keyphrase Extraction".

    Usage example:
    --------------

    >>> import spacy

    >>> nlp = spacy.load("en_core_web_sm")
    >>> nlp.add_pipe(TopicRank(nlp))

    >>> doc = nlp(
        "Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence "
        "concerned with the interactions between computers and human language, in particular how to program computers "
        "to process and analyze large amounts of natural language data. "
    )

    >>> doc._.extract_keywords(n=5)
    """

    defaults: Dict[str, Any] = {
        "clustering_method": "average",
        "distance_metric": "jaccard",
        "threshold": 0.74,
        "alpha": 0.85,
        "tol": 1.0e-6,
        "heuristic": None,
        "candidate_selection": "chunk",
    }

    def candidate_selection(self, doc: Doc) -> Iterable[Candidate]:
        """Get keywords candidates.

        Args:
            doc (Doc): doc.

        Returns:
            Iterable[Candidate]
        """
        C = super().candidate_selection(doc)
        C.sort(key=lambda x: x.lexical_form)
        return C

    def candidate_weighting(self, doc: Doc) -> List[Tuple[Candidate, float]]:
        """Compute the weighted score of each keyword candidate.

        Args:
            doc (Doc): doc.

        Returns:
            list of tuples, candidate with a score.
        """
        res = []
        C = doc._.kw_candidates
        G = self.build_graph(doc)
        W = nx.pagerank_scipy(G, alpha=self.cfg["alpha"], tol=self.cfg["tol"], weight="weight")

        for i, topic in nx.get_node_attributes(G, "C").items():
            offsets = [C[t].offsets[0] for t in topic]
            if self.cfg["heuristic"] == "frequent":
                freq = [len(C[t].surface_forms) for t in topic]
                indexes = [j for j, f in enumerate(freq) if f == max(freq)]
                indexes_offsets = [offsets[j] for j in indexes]
                most_frequent = offsets.index(min(indexes_offsets))
                res.append((C[topic[most_frequent]], W[i]))
            else:
                first = offsets.index(min(offsets))
                res.append((C[topic[first]], W[i]))

        res.sort(key=lambda x: x[1], reverse=True)
        return res

    def build_graph(self, doc: Doc) -> nx.Graph:
        """Build topic graph from candidates.

        Args:
            doc (Doc): doc.

        Returns:
            nx.Graph
        """
        G = nx.Graph()
        C = doc._.kw_candidates
        T = self.topic_clustering(
            doc,
            clustering_method=self.cfg["clustering_method"],
            distance_metric=self.cfg["distance_metric"],
            threshold=self.cfg["threshold"],
        )
        n_topics = len(T)
        for i, j in itertools.combinations(range(n_topics), 2):
            if not G.has_node(i):
                G.add_node(i, C=T[i])
            if not G.has_node(j):
                G.add_node(j, C=T[j])
            G.add_edge(i, j, weight=0.0)
            for c_i in T[i]:
                for c_j in T[j]:
                    for p_i in C[c_i].offsets:
                        for p_j in C[c_j].offsets:
                            gap = abs(p_i - p_j)
                            if p_i < p_j:
                                gap -= len(C[c_i].lexical_form) - 1
                            if p_j < p_i:
                                gap -= len(C[c_j].lexical_form) - 1
                            G[i][j]["weight"] += 1.0 / gap
        return G

    def topic_clustering(
        self, doc: Doc, clustering_method="average", distance_metric="jaccard", threshold=0.74
    ) -> List[List[int]]:
        """Get a list of topics by clustering candidates' lexical forms.

        Args:
            doc (Doc): doc.
            clustering_method (str): optional, clustering method for linkage. default is "average".
            distance_metric (str): optional, distance metric for similarity. default is "jaccard".
            threshold (float): the minimum similarity for clustering, defaults to 0.74,
                               i.e. more than 1/4 of lexical form overlap similarity.

        Returns:
            List of topics, each represented by a list of candidate indices.
        """
        topics = []
        C = doc._.kw_candidates
        if len(C) == 1:
            topics.append([0])
        else:
            X = self.embed(C)
            Y = pdist(X, distance_metric)
            Z = linkage(Y, method=clustering_method)
            clusters = fcluster(Z, t=threshold, criterion="distance")
            for cluster_id in range(1, max(clusters) + 1):
                topics.append([j for j in range(len(clusters)) if clusters[j] == cluster_id])
        return topics

    @staticmethod
    def embed(candidates: List[Candidate]) -> np.ndarray:
        """Get embedding matrix of candidates.

        Args:
            candidates (list): List of candidates.

        Returns:
            np.ndarray.
        """
        dim = set()
        for candidate in candidates:
            for w in candidate.lexical_form:
                dim.add(w)
        dim = list(dim)
        X = np.zeros((len(candidates), len(dim)))
        for i, c in enumerate(candidates):
            for w in c.lexical_form:
                X[i, dim.index(w)] += 1
        return X


if __name__ == "__main__":
    import spacy

    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe(TopicRank(nlp))

    doc = nlp(
        "Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence "
        "concerned with the interactions between computers and human language, in particular how to program computers "
        "to process and analyze large amounts of natural language data. "
    )

    for keyword, score in doc._.extract_keywords(n=10):
        print(keyword, "-", score)
