import os
from typing import List

import numpy as np
import torch
from tqdm import tqdm
from vieneu import Vieneu


class SpeakerViet:
    def __init__(
        self, output_dir="../vietnamse_translations/audio", batch_size=8, device=None
    ):
        self.output_dir = output_dir
        self.batch_size = batch_size

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        print(f"Loading VieNeu TTS model on {device}...")

        if device.startswith("cuda"):
            self.tts = Vieneu(
                backbone_repo="pnnbao-ump/VieNeu-TTS-0.3B",
                backbone_device=device,
                codec_repo="neuphonic/neucodec",
                codec_device=device,
            )
        else:
            self.tts = Vieneu()

        available_voices = self.tts.list_preset_voices()
        if available_voices:
            _, voice_id = (
                available_voices[1]
                if len(available_voices) > 1
                else available_voices[0]
            )
            self.voice = self.tts.get_preset_voice(voice_id)
            print(f"Using voice: {voice_id}")
        else:
            self.voice = None

    def _generate_filename(self, text: str, index: int) -> str:
        safe_text = "".join(c if c.isalnum() else "_" for c in text[:20])
        return f"{index:06d}_{safe_text}.wav"

    def synthesize_batch(self, phrases: List[str], start_index: int) -> List[str]:
        paths = []
        for i, phrase in enumerate(phrases):
            idx = start_index + i
            filename = self._generate_filename(phrase, idx)
            filepath = os.path.join(self.output_dir, filename)

            if self.voice:
                audio = self.tts.infer(text=phrase, voice=self.voice)
            else:
                audio = self.tts.infer(text=phrase)

            self.tts.save(audio, filepath)
            paths.append(filepath)

        return paths

    def synthesize(self, phrases: np.ndarray) -> np.ndarray:
        original_shape = phrases.shape
        phrases_list = phrases.flatten().tolist()

        all_paths = []

        with tqdm(
            total=len(phrases_list),
            desc="Generating Vietnamese audio",
            unit="phrase",
            ncols=100,
        ) as pbar:
            for i in range(0, len(phrases_list), self.batch_size):
                batch = phrases_list[i : i + self.batch_size]
                paths = self.synthesize_batch(batch, start_index=i)
                all_paths.extend(paths)
                pbar.update(len(batch))

        paths_array = np.array(all_paths).reshape(original_shape)
        return paths_array

    def close(self):
        self.tts.close()
