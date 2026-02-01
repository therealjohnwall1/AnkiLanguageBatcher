import os

import numpy as np


def pull_language_files(lang="vi") -> tuple[np.ndarray, np.ndarray]:
    """
    See README.txt in top-open-subtitles-sentences for more information.
    """

    data_dir = os.path.join("../", "top-open-subtitles-sentences", "bld")

    top_words_path = os.path.join(data_dir, "top_words", f"{lang}_top_words.csv")
    top_sentences_path = os.path.join(
        data_dir, "top_sentences", f"{lang}_top_sentences.csv"
    )

    def read_words(p):
        with open(p) as f:
            next(f)
            return np.array([line.split(",")[0] for line in f])

    return read_words(top_words_path), read_words(top_sentences_path)


if __name__ == "__main__":
    pull_language_files()
