from pathlib import Path

import numtorch as nt
import numpy as np

np.random.seed(0)


class RNN(nt.layers.Layer):
    def __init__(self, num_inputs, num_hidden, num_outputs, activation=None):
        if activation is None:
            raise Exception("Activation layer expected")

        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        self.activation = activation

        self.weight_input_hidden = nt.layers.LinearLayer(
            num_inputs, num_hidden)
        self.weight_hidden_hidden = nt.layers.LinearLayer(
            num_hidden, num_hidden)
        self.weight_hidden_output = nt.layers.LinearLayer(
            num_hidden, num_outputs)

        self.params = self.weight_input_hidden.get_params()
        self.params += self.weight_hidden_hidden.get_params()
        self.params += self.weight_hidden_output.get_params()

    def init_hidden(self, batch_size=1):
        return nt.Tensor(np.zeros((batch_size, self.num_hidden)), {"autograd": True})

    def get_params(self):
        return self.params

    def forward(self, input, hidden):
        hidden_hidden = self.weight_hidden_hidden.forward(hidden)
        input_hidden = self.weight_input_hidden.forward(input)
        next_hidden = self.activation.forward(input_hidden + hidden_hidden)
        hidden_output = self.weight_hidden_output.forward(next_hidden)

        return hidden_output, next_hidden


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
model = RNN(16, 16, len(vocab), activation=nt.layers.Sigmoid())

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
