import re
import numpy as np

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Tuple, Any, List
from spacy.tokens.doc import Doc

from spacy_ke.base import KeywordExtractor
from spacy_ke.util import Candidate


@dataclass
class YakeFeatures:
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


class Yake(KeywordExtractor):
    """spaCy implementation of "YAKE! Collection-Independent Automatic Keyword Extractor".

    Usage example:
    --------------

    >>> import spacy

    >>> nlp = spacy.load("en_core_web_sm")
    >>> nlp.add_pipe(Yake(nlp))

    >>> doc = nlp(
        "Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence "
        "concerned with the interactions between computers and human language, in particular how to program computers "
        "to process and analyze large amounts of natural language data. "
    )

    >>> doc._.extract_keywords(n=5)
    """

    defaults: Dict[str, Any] = {"window": 2, "lemmatize": False, "candidate_selection": "ngram"}

    def candidate_weighting(self, doc: Doc) -> List[Tuple[Candidate, Any]]:
        """Compute the weighted score of each keyword candidate.

        Args:
            doc (Doc): doc.

        Returns:
            list of tuples, candidate with a score.
        """
        res = []
        vocab = self._build_vocab_features(doc)
        for candidate in doc._.kw_candidates:
            if self.cfg["lemmatize"]:
                n_offsets = len(candidate.offsets)
                weights = [vocab[w].weight for w in candidate.lexical_form]
                candidate_w = np.prod(weights) / (n_offsets * (1 + sum(weights)))
                res.append((candidate, candidate_w))
            else:
                lowercase_forms = [
                    " ".join(t.text.lower() for t in sf) for sf in candidate.surface_forms
                ]
                for i, sf in enumerate(candidate.surface_forms):
                    tf = lowercase_forms.count(sf.text.lower())
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
        res.sort(key=lambda x: x[1])
        return res

    def _build_vocab_features(self, doc: Doc) -> Dict[str, YakeFeatures]:
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
            features[word] = YakeFeatures(
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
        for sent_i, sent in enumerate(doc.sents):
            for token in sent:
                if token.is_alpha and not re.search("(?i)^-[lr][rcs]b-$", token.text):
                    if lemmatize:
                        word = token.lemma_.lower()
                    else:
                        word = token.lower_
                    words[word].add((sent.start + token.i, sent.start, sent_i, token.text))
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
        for sent_i, sent in enumerate(doc.sents):
            block = []
            for token in sent:
                word = token.lower_
                if word not in vocab:
                    block = []
                    continue
                # add the left context
                contexts[word][0].extend(
                    [w for w in block[max(0, len(block) - window) : len(block)]]
                )
                # add the right context
                for w in block[max(0, len(block) - window) : len(block)]:
                    contexts[w][1].append(word)
                # add word to the current block
                block.append(word)
        return contexts
