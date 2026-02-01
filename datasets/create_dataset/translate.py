from typing import List

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


class Translate:
    def __init__(self, model_name="tencent/HY-MT1.5-7B", batch_size=8):
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading model on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
            low_cpu_mem_usage=True,
        )

    def translate_batch(
        self, phrases: List[str], src_lang="Vietnamese", tgt_lang="English"
    ) -> List[str]:
        all_prompts = []
        for phrase in phrases:
            prompt = f"Translate the following segment into {tgt_lang}, without additional explanation.\n\n{phrase}"
            all_prompts.append(prompt)

        formatted_prompts = []
        for prompt in all_prompts:
            messages = [{"role": "user", "content": prompt}]
            formatted = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            formatted_prompts.append(formatted)

        self.tokenizer.padding_side = "left"

        encoded = self.tokenizer(
            formatted_prompts,
            return_tensors="pt",
            padding=True,
            truncation=False,
            add_special_tokens=False,
        )

        input_ids = encoded["input_ids"].to(self.model.device)
        attention_mask = encoded["attention_mask"].to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=2048,
                num_beams=5,
                early_stopping=True,
                top_k=20,
                top_p=0.6,
                repetition_penalty=1.05,
                temperature=0.7,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        translations = []
        for output in outputs:
            decoded = self.tokenizer.decode(output, skip_special_tokens=True)
            if "without additional explanation." in decoded:
                translation = decoded.split("without additional explanation.")[
                    -1
                ].strip()
            else:
                translation = decoded.strip()
            translations.append(translation)

        return translations

    def translate(
        self, phrases: np.ndarray, src_lang="Vietnamese", tgt_lang="English"
    ) -> np.ndarray:
        original_shape = phrases.shape
        phrases_list = phrases.flatten().tolist()

        all_translations = []

        with tqdm(
            total=len(phrases_list),
            desc=f"Translating {src_lang}→{tgt_lang}",
            unit="phrase",
            ncols=100,
        ) as pbar:
            for i in range(0, len(phrases_list), self.batch_size):
                batch = phrases_list[i : i + self.batch_size]
                translations = self.translate_batch(batch, src_lang, tgt_lang)
                all_translations.extend(translations)
                pbar.update(len(batch))

        translations_array = np.array(all_translations).reshape(original_shape)
        return translations_array
