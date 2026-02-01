from typing import List

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class Translate:
    def __init__(self, model_name="tencent/HY-MT1.5-7B", batch_size=8):
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading model on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )

    def translate_batch(
        self, phrases: List[str], src_lang="Vietnamese", tgt_lang="English"
    ) -> List[str]:
        all_messages = []
        for phrase in phrases:
            messages = [
                {
                    "role": "user",
                    "content": f"Translate the following segment into {tgt_lang}, without additional explanation.\n\n{phrase}",
                }
            ]
            all_messages.append(messages)

        tokenized_inputs = []
        for messages in all_messages:
            tokenized = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                return_tensors="pt",
            )
            tokenized_inputs.append(tokenized)

        # Pad to same length
        max_len = max(t.shape[1] for t in tokenized_inputs)
        padded_inputs = []
        attention_masks = []

        for tokenized in tokenized_inputs:
            pad_len = max_len - tokenized.shape[1]
            if pad_len > 0:
                padding = torch.full(
                    (1, pad_len),
                    self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                    dtype=tokenized.dtype,
                )
                padded = torch.cat([tokenized, padding], dim=1)
                mask = torch.cat(
                    [torch.ones_like(tokenized), torch.zeros_like(padding)], dim=1
                )
            else:
                padded = tokenized
                mask = torch.ones_like(tokenized)

            padded_inputs.append(padded)
            attention_masks.append(mask)

        batch_input_ids = torch.cat(padded_inputs, dim=0).to(self.model.device)
        batch_attention_mask = torch.cat(attention_masks, dim=0).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                batch_input_ids,
                attention_mask=batch_attention_mask,
                max_new_tokens=2048,
                num_beams=5,
                early_stopping=True,
                top_k=20,
                top_p=0.6,
                repetition_penalty=1.05,
                temperature=0.7,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
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
        for i in range(0, len(phrases_list), self.batch_size):
            batch = phrases_list[i : i + self.batch_size]
            translations = self.translate_batch(batch, src_lang, tgt_lang)
            all_translations.extend(translations)

        translations_array = np.array(all_translations).reshape(original_shape)
        return translations_array
