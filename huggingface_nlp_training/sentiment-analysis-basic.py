from transformers import pipeline, AutoTokenizer

if __name__ == '__main__':
    classifier = pipeline("sentiment-analysis")
    text = "I've been waiting for a HuggingFace course my whole life."
    result = classifier(text)
    print(result)

    text_list = ["I've been waiting for a HuggingFace course my whole life.", "I hate this so much!"]
    res = classifier(text_list)
    print(res)

    checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    raw_inputs = [
        "I've been waiting for a HuggingFace course my whole life.",
        "I hate this so much!",
    ]
    inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
    print(inputs)