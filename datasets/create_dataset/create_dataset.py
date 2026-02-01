import os

import numpy as np
from files import pull_language_files
from translate import Translate


def create_dataset(save_path="../vietnamse_translations", batch_size=8):
    top_words_f, top_sentences_f = pull_language_files()
    slater = Translate(batch_size=batch_size)  # change on gpu

    top_words_l = slater.translate(top_words_f)
    top_sentences_l = slater.translate(top_sentences_f)

    words = np.column_stack((top_words_f, top_words_l))
    sentences = np.column_stack((top_sentences_f, top_sentences_l))
    print(words.shape, sentences.shape)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    np.save(os.path.join(save_path, "words"), words)
    np.save(os.path.join(save_path, "sentences"), sentences)


def save_txt(save_path="../vietnamse_translations"):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    x = np.load(os.path.join(save_path, "words.npy"))
    np.savetxt(os.path.join(save_path, "words.txt"), x, fmt="%s")

    x = np.load(os.path.join(save_path, "sentences.npy"))
    np.savetxt(os.path.join(save_path, "sentences.txt"), x, fmt="%s")


if __name__ == "__main__":
    create_dataset(batch_size=32)
    save_txt()
