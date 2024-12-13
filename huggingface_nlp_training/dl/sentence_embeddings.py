from transformers.utils import logging
from sentence_transformers import SentenceTransformer
from sentence_transformers import util

logging.set_verbosity_error()

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences1 = ['The cat sits outside',
              'A man is playing guitar',
              'The movies are awesome']


embeddings1 = model.encode(sentences1, convert_to_tensor=True)

sentences2 = ['The dog plays in the garden',
              'A woman watches TV',
              'The new movie is so great']

embeddings2 = model.encode(sentences2,
                           convert_to_tensor=True)

cosine_scores = util.cos_sim(embeddings1,embeddings2)

print(cosine_scores)

for i in range(len(sentences1)):
    print("{} \t\t {} \t\t Score: {:.4f}".format(sentences1[i],
                                                 sentences2[i],
                                                 cosine_scores[i][i]))



