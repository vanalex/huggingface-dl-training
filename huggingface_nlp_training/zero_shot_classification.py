from transformers import pipeline


if __name__ == '__main__':
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    result = classifier(
    "This is a course about the Transformers library",
            candidate_labels=["education", "politics", "business"],
    )
    print(result)

    result = classifier("I have a problem with my iphone that needs to be resolved asap!",
         candidate_labels=["urgent", "not urgent", "phone", "tablet", "computer"],
         )
    print(result)