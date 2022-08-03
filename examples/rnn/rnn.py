from pathlib import Path

import numtorch as nt
import numpy as np

np.random.seed(0)

f = open(Path(__file__).parent / "./qa1_single-supporting-fact_train.txt", "r")
raw = f.readlines()
f.close()

tokens = list()
for line in raw[0:1000]:
    tokens.append(line.lower().replace("\n", "").split(" ")[1:])

new_tokens = list()
for line in tokens:
    new_tokens.append(['-'] * (6 - len(line)) + line)
tokens = new_tokens

vocab = set()
for sent in tokens:
    for word in sent:
        vocab.add(word)

vocab = list(vocab)

word2index = {}
for i, word in enumerate(vocab):
    word2index[word] = i


def words2indices(sentence):
    idx = list()

    for word in sentence:
        idx.append(word2index[word])

    return idx


indices = list()
for line in tokens:
    idx = list()

    for w in line:
        idx.append(word2index[w])

    indices.append(idx)

data = np.array(indices)

embed = nt.layers.EmbeddingLayer(len(vocab), 16)
model = nt.layers.RecurrentCell(
    16, 16, len(vocab), activation=nt.layers.Sigmoid())

criterion = nt.layers.CrossEntropyLoss()

params = model.get_params() + embed.get_params()
optimizer = nt.optimizers.SGDOptimizer(params=params, lr=0.05)

for iter in range(1000):
    batch_size = 100
    total_loss = 0

    hidden = model.init_hidden(batch_size=batch_size)

    for t in range(5):
        input = nt.Tensor(data[0:batch_size, t], {"autograd": True})
        rnn_input = embed.forward(input=input)
        output, hidden = model.forward(input=rnn_input, hidden=hidden)

    target = nt.Tensor(data[0:batch_size, t+1], {"autograd": True})
    loss = criterion.forward(output, target)
    loss.backward()
    optimizer.step()
    total_loss += loss.data
    if(iter % 200 == 0):
        p_correct = (target.data == np.argmax(output.data, axis=1)).mean()
        print("Loss:", total_loss / (len(data)/batch_size),
              "% Correct:", p_correct)

batch_size = 1
hidden = model.init_hidden(batch_size=batch_size)
for t in range(5):
    input = nt.Tensor(data[0:batch_size, t], {"autograd": True})
    rnn_input = embed.forward(input=input)
    output, hidden = model.forward(input=rnn_input, hidden=hidden)

target = nt. Tensor(data[0:batch_size, t+1], {"autograd": True})
loss = criterion.forward(output, target)

ctx = ""
for idx in data[0:batch_size][0][0:-1]:
    ctx += vocab[idx] + " "

print("Context:", ctx)
print("True:", vocab[target.data[0]])
print("Pred:", vocab[output.data.argmax()])
