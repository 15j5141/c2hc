from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import json


class CreateDocModel:
    def __init__(self):
        self.pathModel = "./in/doc2vec.model"
        self.model = Doc2Vec(
            [TaggedDocument(doc, [i]) for i, doc in enumerate([["RET"]])],
            vector_size=2,
            window=5,
            min_count=1,
            workers=4,
        )
        with open("./out/results.csv", "r") as f:
            self.lines = []
            lines = f.readlines()
            for line in lines:
                self.lines.append(line.split(","))

    def load(self):
        self.model = Doc2Vec.load(self.pathModel)

    def save(self):
        self.model.save(self.pathModel)

    def fitAll(self, docs=[[]]):
        documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(docs)]
        self.model = Doc2Vec(
            documents,
            vector_size=100,
            window=5,
            min_count=1,
            workers=4,
            dm=1,
            epochs=100,
        )

    def fitResults(self):
        self.fitAll(self.lines)


if __name__ == "__main__":
    models = CreateDocModel()
    models.fitResults()
    models.save()
    print(models.model.infer_vector(["RET"]))
