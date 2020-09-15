import re
import numpy as np
import editdistance

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterator, Tuple, Any, List

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


@dataclass
class Features:
    """ term frequency """

    tf: float

    """
    1. CASING: gives importance to acronyms or words starting with a capital letter.
               CASING(w) = max(TF(U(w)), TF(A(w))) / (1 + log(TF(w)))
               with TF(U(w) being the # times the word starts with an uppercase
               letter, excepts beginning of sentences. TF(A(w)) is the # times
               the word is marked as an acronym.
    """
    casing: float

    """
    2. POSITION: gives importance to words occurring at the beginning of the document.
               POSITION(w) = log( log( 3 + Median(Sen(w)) ) )
               with Sen(w) contains the position of the sentences where w
               occurs.
    """

    position: float

    """
    3. FREQUENCY: gives importance to frequent words.
               FREQUENCY(w) = TF(w) / ( MEAN_TF + STD_TF)
               with MEAN_TF and STD_TF computed on valid_tfs which are words
               that are not stopwords.
    """
    frequency: float

    """
    4. RELATEDNESS: gives importance to words that do not have the characteristics of stopwords.
               RELATEDNESS(w) = 1 + (WR+WL)*(TF(w)/MAX_TF) + PL + PR
    """
    relatedness: float

    """
    5. DIFFERENT: gives importance to words that occurs in multiple sentences.
               DIFFERENT(w) = SF(w) / # sentences
               with SF(w) being the sentence frequency of word w.
    """
    different: float

    """ right and left context words """
    ctx: Tuple[List[str], List[str]]

    @property
    def weight(self):
        A = self.casing
        B = self.position
        C = self.frequency
        D = self.relatedness
        E = self.different

        return (D * B) / (A + (C / D) + (E / D))


@component(
    "yake",
    requires=["token.pos", "token.dep", "doc.sents"],
    assigns=["doc._.kw", "doc._.kw_vocab", "doc._.kw_candidates"],
)
class Yake:
    cfg: Dict[str, Any] = {
        "window": 2,
        "ngram": 3,
        "lemmatize": False,
        "min_distance": 0.5,
    }

    def __init__(self, nlp: Language, **overrides):
        self.nlp = nlp
        self.cfg.update(overrides)
        self.first_run = True

    def __call__(self, doc: Doc) -> Doc:
        if self.first_run:
            self.init_component()
            self.first_run = False

        return doc

    def init_component(self):
        Doc.set_extension("kw", method=self.kw)
        Doc.set_extension("kw_vocab", getter=self.kw_vocab)
        Doc.set_extension("kw_candidates", getter=self.kw_candidates)

    def kw(self, doc: Doc, n=10, similarity_thresh=0.45):
        """Returns the n-best candidates given the weights.

        Args:
            doc (Doc): doc.
            n (int): the number of candidates, defaults to 10.
            similarity_thresh (float): optional, similarity thresh for dedup. default is 0.45.

        Returns:
            List
        """
        res = []
        candidates_weighted = doc._.kw_candidates
        candidates_weighted.sort(key=lambda x: x[1])
        for candidate, candidate_w in candidates_weighted:
            if similarity_thresh > 0.0:
                redundant = False
                for prev_candidate, _ in res:
                    if candidate.similarity(prev_candidate) > similarity_thresh:
                        redundant = True
                        break
                if redundant:
                    continue
            res.append((candidate, candidate_w))
            if len(res) >= n:
                break
        res = res[: min(n, len(res))]
        res = [(c.surface_forms[0], score) for c, score in res]
        return res

    def kw_candidates(self, doc: Doc):
        """Compute the weighted score of each candidate.

        Args:
            doc (Doc): doc.

        Returns:
            dict, mapping of each candidate lemma to
        """
        res = []
        vocab = doc._.kw_vocab
        candidates = dict()

        for sentence_id, offset, ngram_span in self._ngrams(doc, n=self.cfg["ngram"]):
            if not self._is_candidate(ngram_span):
                continue
            idx = ngram_span.lemma_
            try:
                c = candidates[idx]
            except (KeyError, IndexError):
                lexical_form = [token.lemma_.lower() for token in ngram_span]
                c = candidates[idx] = Candidate(lexical_form, [], [], [])
            c.surface_forms.append(ngram_span)
            c.offsets.append(offset)
            c.sentence_ids.append(sentence_id)

        for idx, candidate in candidates.items():
            if self.cfg["lemmatize"]:
                n_offsets = len(candidate.offsets)
                weights = [vocab[w].weight for w in candidate.lexical_form]
                candidate_w = np.prod(weights) / (n_offsets * (1 + sum(weights)))
                res.append((candidate, candidate_w))
            else:
                lowercase_forms = [
                    " ".join(t.lower_ for t in sf) for sf in candidate.surface_forms
                ]
                for i, sf in enumerate(candidate.surface_forms):
                    tf = lowercase_forms.count(sf.lower_)
                    prod_ = 1.0
                    sum_ = 0.0
                    for j, token in enumerate(sf):
                        if token.is_stop:
                            term_stop = token.lower_
                            prob_t1 = prob_t2 = 0
                            if j - 1 >= 0:
                                left = sf[j - 1]
                                prob_t1 = (
                                        vocab[left.lower_].ctx[1].count(term_stop)
                                        / vocab[left.lower_].tf
                                )
                            if j + 1 < len(sf):
                                right = sf[j + 1]
                                prob_t2 = (
                                        vocab[term_stop].ctx[0].count(right.text)
                                        / vocab[right.lower_].tf
                                )
                            prob = prob_t1 * prob_t2
                            prod_ *= 1 + (1 - prob)
                            sum_ -= 1 - prob
                        else:
                            prod_ *= vocab[token.lower_].weight
                            sum_ += vocab[token.lower_].weight
                    if sum_ == -1:
                        # The candidate is a one token stopword at the start or
                        #  the end of the sentence
                        # Setting sum_ to -1+eps so 1+sum_ != 0
                        sum_ = -0.99999999999
                    candidate_w = prod_
                    candidate_w /= tf * (1 + sum_)
                    res.append((candidate, candidate_w))
        return res

    def kw_vocab(self, doc: Doc) -> Dict[str, Features]:
        """Compute the weight of individual words using yake features (see ``Features`` class).

        Args:
            doc (Doc): doc.

        Returns:
            dict[word<str> -> features<Features>]
        """
        features = dict()
        vocab = self._build_vocab(doc, lemmatize=self.cfg["lemmatize"])
        contexts = self._build_ctx(doc, vocab, window=self.cfg["window"])
        n_sentences = len(list(doc.sents))
        stop_words = self.nlp.Defaults.stop_words
        freqs = [len(vocab[w]) for w in vocab]
        freqs_nsw = [len(vocab[w]) for w in vocab if stop_words and w not in stop_words]
        mean_freq = np.mean(freqs_nsw)
        std_freq = np.std(freqs_nsw)
        max_freq = np.max(freqs)

        for word in vocab:
            # Term Frequency
            #   & Uppercase/Acronym Term Frequencies
            tf = len(vocab[word])
            tf_a = 0
            tf_u = 0

            for (offset, shift, sent_id, surface_form) in vocab[word]:
                if surface_form.isupper() and len(word) > 1:
                    tf_a += 1
                elif surface_form[0].isupper() and offset != shift:
                    tf_u += 1

            # 1. CASING
            casing = max(tf_a, tf_u) / (1.0 + np.log(tf))

            # 2. POSITION
            sentence_ids = list(set([t[2] for t in vocab[word]]))
            position = np.log(3.0 + np.median(sentence_ids))
            position = np.log(position)

            # 3. FREQUENCY
            frequency = tf
            frequency /= mean_freq + std_freq

            # 4. RELATEDNESS
            wl = 0.0
            if len(contexts[word][0]):
                wl = len(set(contexts[word][0]))
                wl /= len(contexts[word][0])
            pl = len(set(contexts[word][0])) / max_freq

            wr = 0.0
            if len(contexts[word][1]):
                wr = len(set(contexts[word][1]))
                wr /= len(contexts[word][1])
            pr = len(set(contexts[word][1])) / max_freq

            relatedness = 1
            relatedness += (wr + wl) * (tf / max_freq)
            relatedness += pl
            relatedness += pr

            # 5. DIFFERENT
            different = len(sentence_ids) / n_sentences
            features[word] = Features(
                tf, casing, position, frequency, relatedness, different, contexts[word]
            )

        return features

    @staticmethod
    def _build_vocab(doc: Doc, lemmatize: bool = False) -> Dict[str, Tuple[int, int, str]]:
        """Build the vocabulary that will be used to weight candidates.
        Only words containing at least one alpha-numeric character are kept.

        Args:
            doc (Doc): doc.
            lemmatize (bool): whether or not to lemmatize vocabulary.

        Returns:
            dict.
        """
        words = defaultdict(set)
        sentences = list(doc.sents)

        for i, sentence in enumerate(sentences):
            shift = sum([len(s) for s in sentences[0:i]])
            for word in sentence:
                if word.is_alpha and not re.search("(?i)^-[lr][rcs]b-$", word.text):
                    if lemmatize:
                        word_idx = word.lemma_.lower()
                    else:
                        word_idx = word.lower_
                    words[word_idx].add((shift + word.i, shift, i, word.text))

        return words

    @staticmethod
    def _build_ctx(doc: Doc, vocab: Dict, window: int = 2) -> Dict[str, Tuple[List, List]]:
        """Build the contexts of the words for computing the relatedness
        feature. Words that occur within a window of n words are considered as
        context words. Only words co-occurring in a block (sequence of words
        that appear in the vocabulary) are considered.

        Args:
            doc (Doc): doc.
            vocab (Dict): See ``_vocab()`` method.
            window (int): optional, size of context window, default is 2.

        Returns:
            dict.
        """
        contexts = defaultdict(lambda: ([], []))
        for i, sentence in enumerate(doc.sents):
            words = [w.lower_ for w in sentence]
            block = []
            for j, word in enumerate(words):
                if word not in vocab:
                    block = []
                    continue
                # add the left context
                contexts[word][0].extend(
                    [w for w in block[max(0, len(block) - window): len(block)]]
                )
                # add the right context
                for w in block[max(0, len(block) - window): len(block)]:
                    contexts[w][1].append(word)
                # add word to the current block
                block.append(word)
        return contexts

    @staticmethod
    def _ngrams(doc: Doc, n=3) -> Iterator[Tuple[int, int, Span]]:
        """Select all the n-grams and populate the candidate container.

        Args:
            doc (Doc): doc.
            n (int): the n-gram length, defaults to 3.

        Returns:
            Iterator(sentence_id<int>, offset<int>, ngram<Span>)
        """
        sentences = list(doc.sents)
        for sentence_id, sentence in enumerate(sentences):
            sentence_length = len(sentence)
            skip = min(n, sentence_length)
            shift = sum([len(s) for s in sentences[0:sentence_id]])
            for j in range(sentence_length):
                for k in range(j + 1, min(j + 1 + skip, sentence_length + 1)):
                    tokens = sentence[j:k]
                    offset = shift + j
                    yield sentence_id, offset, tokens

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
        return True
