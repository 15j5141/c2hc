from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import json


class CreateDocModel:
    def __init__(self):
        self.pathModel = "./in/doc2vec.model"
        self.model = Doc2Vec([], vector_size=2, window=5, min_count=1, workers=4)
        with open("./records.json", "r") as f:
            self.records: list = json.load(f)

    def load(self):
        self.model = Doc2Vec.load(self.pathModel)

    def save(self):
        self.model.save(self.pathModel)

    def fitAll(self, docs=[[]]):
        documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(docs)]
        self.model = Doc2Vec(documents, vector_size=2, window=5, min_count=1, workers=4)

    def fitRecords(self):
        for j in range(len(self.records)):
            COMPILE_FILE = "in/Project_CodeNet/" + self.records[j]["c"]
            INPUT_TEXT = self.records[j]["in"]
