from spacy.language import Language
from thinc.api import Config

from spacy_ke.about import *
from spacy_ke.yake import Yake
from spacy_ke.textrank import TextRank
from spacy_ke.positionrank import PositionRank
from spacy_ke.topicrank import TopicRank

if hasattr(Language, "factory"):
    defaults = Config().from_str(
        """
    [Yake]
    window = 2
    lemmatize = false
    candidate_selection = ngram
    
    [TextRank]
    window = 3
    alpha = 0.85
    tol = 1.0e-6
    candidate_selection = chunk
    
    [PositionRank]
    window = 10
    alpha = 0.85
    tol = 1.0e-5
    normalize = false
    candidate_selection = chunk
    
    [TopicRank]
    clustering_method = average
    distance_metric = jaccard
    threshold = 0.74
    alpha = 0.85
    tol = 1.0e-6
    heuristic = null
    candidate_selection = chunk
    """
    )

    @Language.factory("yake", default_config=defaults["Yake"])
    def make_yake(
        nlp: Language, name: str, window: int, lemmatize: bool, candidate_selection: str
    ):
        return Yake(
            nlp,
            window=window,
            lemmatize=lemmatize,
            candidate_selection=candidate_selection,
        )

    @Language.factory("textrank", default_config=defaults["TextRank"])
    def make_textrank(
        nlp: Language,
        name: str,
        window: int,
        alpha: float,
        tol: float,
        candidate_selection: str,
    ):
        return TextRank(
            nlp,
            window=window,
            alpha=alpha,
            tol=tol,
            candidate_selection=candidate_selection,
        )

    @Language.factory("positionrank", default_config=defaults["PositionRank"])
    def make_positionrank(
        nlp: Language,
        name: str,
        window: int,
        alpha: float,
        tol: float,
        normalize: bool,
        candidate_selection: str,
    ):
        return PositionRank(
            nlp,
            window=window,
            alpha=alpha,
            tol=tol,
            normalize=normalize,
            candidate_selection=candidate_selection,
        )

    @Language.factory("topicrank", default_config=defaults["TopicRank"])
    def make_topicrank(
        nlp: Language,
        name: str,
        clustering_method: str,
        distance_metric: str,
        threshold: float,
        alpha: float,
        tol: float,
        heuristic: str,
        candidate_selection: str,
    ):
        return TopicRank(
            nlp,
            clustering_method=clustering_method,
            distance_metric=distance_metric,
            threshold=threshold,
            alpha=alpha,
            tol=tol,
            heuristic=heuristic,
            candidate_selection=candidate_selection,
        )
