from transformers import pipeline

class ZeroShotCategoryClassifier:
    def __init__(self):
        self.classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli"
        )

        self.labels = [
            "Fiction",
            "Nonfiction",
            "Fantasy",
            "Science",
            "History",
            "Romance",
            "Mystery"
        ]

    def predict(self, text: str) -> str:
        result = self.classifier(text, self.labels)
        return result["labels"][0]
