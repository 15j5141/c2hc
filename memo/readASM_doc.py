import json
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

# def corpus_to_sentences(corpus):
#     docs   = [read_document(x) for x in corpus]
#     for idx, (doc, name) in enumerate(zip(docs, corpus)):
#         sys.stdout.write('\r前処理中 {}/{}'.format(idx, len(corpus)))
#         yield doc_to_sentence(doc, name)
def ToTaggedDocument(array):
    return list(TaggedDocument(doc, [i]) for i, doc in enumerate(array))


with open("src\\mnemonic.json", "r") as pt:
    json_mnemonic = json.load(pt)
    vocabulary = ToTaggedDocument([json_mnemonic])

with open("src\\results.json", "r") as pt:
    json_results = json.load(pt)


sentences = [
    json_results,
    ["ADD", "RET"],
]
# sentences = [json_mnemonic,["ADD","RET"],]
documents = ToTaggedDocument(sentences)
print(len(documents))
model = Doc2Vec(
    documents, vector_size=10, window=5, min_count=1, workers=4, dm=0, epochs=100
)
# model.build_vocab(corpus_iterable=vocabulary,)
# model.train(corpus_iterable=documents,total_examples = len(documents), epochs=100)

# print('\n訓練開始')
# print('Epoch: 100', end='')
# for epoch in range(100):
#     print('{}'.format("|"+str(model.get_latest_training_loss())), end="")
#     model.train(corpus_iterable=documents,total_examples = len(documents), epochs=1)
#     model.alpha -= (0.025 - 0.0001) / 19
#     model.min_alpha = model.alpha
print("")
model.save("doc2vec.model")
model = Doc2Vec.load("doc2vec.model")

vector = model.infer_vector([" ".join(json_results)])
print(vector)
print("infer_vector :v")
print(model.infer_vector([" ".join(json_mnemonic)]))
print("dv", model.dv)
print("dv1", model.dv[1])
# print(model.dv.most_similar(positive=[0]))
