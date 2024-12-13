import torch
from transformers.utils import logging
from transformers import pipeline

logging.set_verbosity_error()

chat = [
    {"role": "system", "content": "You are a sassy, wise-cracking robot as imagined by Hollywood circa 1986."},
    {"role": "user", "content": "Hey, can you tell me any fun things to do in New York?"}
]

pipe = pipeline(task="text-generation", model="facebook/blenderbot-400M-distill", torch_dtype=torch.bfloat16, device_map="auto")

response = pipe(chat, max_new_tokens=512)
print(response[0]['generated_text'][-1]['content'])

chat = response[0]['generated_text']
chat.append(
    {"role": "user", "content": "Wait, what's so wild about soup cans?"}
)
response = pipe(chat, max_new_tokens=512)
print(response[0]['generated_text'][-1]['content'])


