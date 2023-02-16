# アセンブリコードの配列jsonをone-hotなベクトルにする.
import os
import json
import re
import csv
import pandas as pd
import numpy as np

# from nptyping import Array

# import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

# import readASM


class SArray2Vec:
    def __init__(self):
        self.path_mnemonic = "src/mnemonic.json"
        self.path_section = "src/section.json"

    def load(self, filename: str = None):
        self.lines = []
        current_dir = os.path.dirname(os.path.abspath(__file__)) + "/"
        path = (current_dir + "out/results.json") if filename is None else filename
        with open(self.path_mnemonic, "r") as pt:
            json_mnemonic: list = json.load(pt)
        with open(self.path_section, "r") as pt:
            json_section: list = json.load(pt)
            json_section = [(str.upper(x)) for x in json_section]
        self.mnemonic_section_list = json_section + json_mnemonic
        with open(path) as f:
            self.lines = json.load(f)
        self.doc2vecModel = Doc2Vec.load("in/doc2vec.model")

    def toVector(self):
        OneHot = OneHotEncoder(sparse=False, dtype=int)
        # print(self.mnemonic_section_list)
        OneHot.set_params(categories=[self.mnemonic_section_list])
        vector = OneHot.fit_transform(pd.DataFrame(self.lines))
        vector2 = np.sum(vector, axis=0)
        # print(vector2)
        return vector2

    def toVector2(self):
        # [["ADD", "RET"]] -> TaggedDocument
        model = self.doc2vecModel
        vector = model.infer_vector(self.lines)
        vector2 = vector
        # print(vector2)
        return vector2

    def save(self, results):
        # with open("out/results_onehot.json", "w") as f:
        # with open("out/mnemonicCounter.json", "w") as f:
        #     json.dump(results, f, indent=4)
        with open("out/mc_append.csv", "a") as f:
            writer = csv.writer(f)
            writer.writerow(results)


def main():
    sArray2Vec = SArray2Vec()
    sArray2Vec.load()
    vector = sArray2Vec.toVector2()
    sArray2Vec.save(vector.tolist())


if __name__ == "__main__":
    main()
