from transformers.utils import logging
from transformers import pipeline
import torch
import gc

logging.set_verbosity_error()

translator = pipeline(task="translation",
                      model="facebook/nllb-200-distilled-600M",
                      torch_dtype=torch.bfloat16)


text = """\
My puppy is adorable, \
Your kitten is cute.
Her panda is friendly.
His llama is thoughtful. \
We all have nice pets!"""

text_translated = translator(text, src_lang="eng_Latn", tgt_lang="fra_Latn")
print('text translated => ', text_translated)

del translator
gc.collect()