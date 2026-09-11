import pickle
from collections import Counter


class Vocabulary:

    def __init__(self, min_freq=5):
        self.min_freq = min_freq

        self.word_to_idx = {}
        self.idx_to_word = {}

        # 0 is reserved for padding
        self.word_to_idx["<PAD>"] = 0

        self.word_to_idx["<UNK>"] = 1
        self.word_to_idx["<START>"] = 2
        self.word_to_idx["<END>"] = 3

        self.idx_to_word = {
            0: "<PAD>",
            1: "<UNK>",
            2: "<START>",
            3: "<END>"
        }

    def build(self, captions):

        counter = Counter()

        # Count every word
        for caption in captions:

            words = caption.split()

            for word in words:
                counter[word] += 1

        # Add words that satisfy minimum frequency
        for word, frequency in counter.items():

            if frequency >= self.min_freq:

                if word not in self.word_to_idx:

                    idx = len(self.word_to_idx)

                    self.word_to_idx[word] = idx
                    self.idx_to_word[idx] = word

    def numericalize(self, caption):

        """
        Convert a caption into a list of integers.
        """

        tokens = []

        for word in caption.split():

            idx = self.word_to_idx.get(
                word,
                self.word_to_idx["<UNK>"]
            )

            tokens.append(idx)

        return tokens

    def decode(self, sequence):

        """
        Convert integer IDs back into words.
        """

        words = []

        for idx in sequence:

            word = self.idx_to_word.get(
                idx,
                "<UNK>"
            )

            words.append(word)

        return " ".join(words)

    def __len__(self):

        return len(self.word_to_idx)

    def save(self, path):

        with open(path, "wb") as file:

            pickle.dump(self, file)

    @staticmethod
    def load(path):

        with open(path, "rb") as file:

            return pickle.load(file)

