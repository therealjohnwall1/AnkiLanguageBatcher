import os

import numpy as np
from files import pull_language_files
from speaker_viet import SpeakerViet
from translate import Translate


def translate_dataset(input_data, save_path, name, batch_size=8):
    translator = Translate(batch_size=batch_size)
    translations = translator.translate(input_data)

    combined = np.column_stack((input_data, translations))

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    output_path = os.path.join(save_path, f"{name}.npy")
    np.save(output_path, combined)

    return output_path


def generate_audio(npy_path, audio_dir, batch_size=8):
    data = np.load(npy_path)
    source_text = data[:, 0]

    if not os.path.exists(audio_dir):
        os.makedirs(audio_dir)

    speaker = SpeakerViet(output_dir=audio_dir, batch_size=batch_size)
    audio_paths = speaker.synthesize(source_text)
    speaker.close()

    combined = np.column_stack((data, audio_paths))
    np.save(npy_path, combined)

    return npy_path


def save_txt(npy_path):
    data = np.load(npy_path)
    txt_path = npy_path.replace(".npy", ".txt")
    np.savetxt(txt_path, data, fmt="%s", delimiter=",")
    return txt_path


if __name__ == "__main__":
    save_path = "../vietnamse_translations"
    audio_dir = os.path.join(save_path, "audio")
    batch_size = 32

    words_data, sentences_data = pull_language_files()

    words_npy = translate_dataset(words_data, save_path, "words", batch_size)
    sentences_npy = translate_dataset(
        sentences_data, save_path, "sentences", batch_size
    )

    # words_npy = "../vietnamse_translations/words.npy"
    # sentences_npy = "../vietnamse_translations/sentences.npy"

    words_npy = generate_audio(words_npy, os.path.join(audio_dir, "words"), batch_size)
    sentences_npy = generate_audio(
        sentences_npy, os.path.join(audio_dir, "sentences"), batch_size
    )

    save_txt(words_npy)
    save_txt(sentences_npy)
