from collections import Counter, defaultdict
import re


def get_stats(vocab):
    """
    Count the frequency of symbol pairs in the vocabulary.
    :param vocab: Dictionary where keys are words (sequences of symbols) and values are frequencies.
    :return: Counter object with pairs of symbols as keys and their frequencies as values.
    """
    pairs = Counter()
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += freq
    return pairs


def merge_vocab(pair, vocab):
    """
    Merge the most frequent pair of symbols in the vocabulary.
    :param pair: The pair of symbols to merge.
    :param vocab: Dictionary where keys are words and values are frequencies.
    :return: Updated vocabulary with the merged pair.
    """
    bigram = re.escape(" ".join(pair))
    pattern = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")
    new_vocab = {}
    for word in vocab:
        new_word = pattern.sub("".join(pair), word)
        new_vocab[new_word] = vocab[word]
    return new_vocab


def bpe_algorithm(cluster_sequences, num_merges=10):
    """
    Implementation of the Byte Pair Encoding (BPE) algorithm.
    :param cluster_sequences: List of sequences (each sequence is a list of cluster labels).
    :param num_merges: Number of merges to perform.
    :return: Learned BPE merges and the final vocabulary.
    """

    vocab = Counter(" ".join(map(str, seq)) for seq in cluster_sequences)

    print("Initial vocabulary:", vocab)
    merges = []

    for _ in range(num_merges):

        pairs = get_stats(vocab)
        if not pairs:
            break

        best_pair = max(pairs, key=pairs.get)
        merges.append(best_pair)

        vocab = merge_vocab(best_pair, vocab)

    return merges, vocab


def apply_bpe(sequence, merges):
    """
    Apply the BPE algorithm to a sequence using learned merges.
    :param sequence: List of cluster labels.
    :param merges: List of learned merges (pairs of symbols).
    :return: Sequence after applying BPE.
    """
    sequence = " ".join(map(str, sequence))
    for merge in merges:
        bigram = re.escape(" ".join(merge))
        pattern = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")
        sequence = pattern.sub("".join(merge), sequence)
    return sequence.split()
