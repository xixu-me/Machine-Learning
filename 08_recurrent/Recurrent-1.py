import torch
import torch.nn as nn
from torch.autograd import Variable

class RNN(nn.Module):
    """ Recurrent neural network - simple
        x -> hidden -> out -> log-softmax
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        raise NotImplementedError

    def forward(self, input, hidden):
        raise NotImplementedError

    def initHidden(self):
        raise NotImplementedError
    

##########################################################
# Text data utilities - character-level
import glob
import unicodedata
import string


all_letters = string.ascii_letters + " .,;'-" # all legal letters
n_letters = len(all_letters)

def unicodeToAscii(s):
    """ Unicode string to plain ASCII, 
        ref. http://stackoverflow.com/a/518232/2809427
    """
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

def readLines(filename):
    """ Read a file and split into lines """
    lines = open(filename).read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

def build_category_lines(data_file='../data/names/*.txt'):
    """ Build the category_lines dictionary, a list of lines per category """
    
    def findFiles(path): return glob.glob(path)

    category_lines = {}
    all_categories = []
    for filename in findFiles(data_file):
        category = filename.split('/')[-1].split('.')[0]
        all_categories.append(category)
        lines = readLines(filename)
        category_lines[category] = lines

    return category_lines, all_categories

def letterToIndex(letter):
    """ Find letter index from all_letters, e.g. "a" = 0 """
    return all_letters.find(letter)

def lineToTensor(line):
    """ Turn a line into a <line_length x 1 x n_letters>,
        ie, an array of one-hot letter vectors
    """
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor
##########################################################


import random
import time
import math


class TextLearner:
    @staticmethod
    def train_epoch(model, category_tensor, line_tensor):
        hidden = model.initHidden()
        optimizer.zero_grad()

        for i in range(line_tensor.size()[0]):
            output, hidden = model(line_tensor[i], hidden)

        loss = criterion(output, category_tensor)
        loss.backward()

        optimizer.step()

        return output, loss.item()

    @staticmethod
    def train(model, n_epochs, print_every, plot_every, learning_rate):
        def categoryFromOutput(output):
            top_n, top_i = output.data.topk(1) # Tensor out of Variable with .data
            category_i = top_i[0][0]
            return all_categories[category_i], category_i

        def randomTrainingPair():
            """ generate a random training pair """
            def randomChoice(l): return l[random.randint(0, len(l) - 1)]

            category = randomChoice(all_categories)
            line = randomChoice(category_lines[category])
            category_tensor = Variable(torch.LongTensor([all_categories.index(category)]))
            line_tensor = Variable(lineToTensor(line))
            return category, line, category_tensor, line_tensor

        def timeSince(since):
            now = time.time()
            s = now - since
            m = math.floor(s / 60)
            s -= m * 60
            return f'{m}m {int(s): >2d}s'
        start = time.time()

        # Keep track of losses for plotting
        current_loss = 0
        all_losses = []
        for epoch in range(1, n_epochs + 1):
            category, line, category_tensor, line_tensor = randomTrainingPair()
            output, loss = TextLearner.train_epoch(model, category_tensor, line_tensor)
            current_loss += loss

            # Print epoch number, loss, name and guess
            if epoch % print_every == 0:
                guess, guess_i = categoryFromOutput(output)
                correct = '✓' if guess == category else '✗ (%s)' % category
                print(f'{epoch: >6d} {epoch / n_epochs * 100:5.1f}% ({timeSince(start)}) {loss:.4f} {current_loss:.2f} {line} / {guess} {correct}')

            # Add current loss avg to list of losses
            if epoch % plot_every == 0:
                all_losses.append(current_loss / plot_every)
                current_loss = 0

        torch.save(model, 'char_rnn_names.pt')
        
    @staticmethod
    def predict_t(line_tensor):
        """ return an output given a line """
        hidden = model.initHidden()

        for i in range(line_tensor.size()[0]):
            output, hidden = model(line_tensor[i], hidden)

        return output

    @staticmethod
    def predict(model, line, n_predictions=3):
        output = TextLearner.predict_t(Variable(lineToTensor(line)))

        # Get top N categories
        topv, topi = output.data.topk(n_predictions, 1, True)
        predictions = []

        print(f'prediction for {line}:')
        for i in range(n_predictions):
            value = topv[0][i]
            category_index = topi[0][i]
            print(f'  ({value:.2f}) {all_categories[category_index]}')
            predictions.append([value, all_categories[category_index]])

        return predictions
    
    @staticmethod
    def eval_all(model, n_predictions=5):
        correct_n = []
        total_n = []
        correct_str = ''
        for ci, category in enumerate(category_lines):
            total_n.append(len(category))
            cni = 0
            for line in category:
                output = TextLearner.predict_t(Variable(lineToTensor(line)))
                _, topi = output.data.topk(n_predictions)
                cc = 0
                for ii in range(n_predictions):
                    if (topi[0][ii] == ci):
                        cc = 1
                        break
                cni += cc
            correct_n.append(cni)
            correct_str += f'{all_categories[ci]}: {cni / total_n[-1] * 100:5.1f}; '
            
        print(f'correct rate (per category): {correct_str}')
        print(f'total correct rate: {sum(correct_n) / sum(total_n) * 100:5.1f}')

    
if __name__ == '__main__':
    print(f'{n_letters} legal letters: {all_letters}')
    category_lines, all_categories = build_category_lines()
    n_categories = len(all_categories)
    print(f'{n_categories} categories: {all_categories}')
    print(f"first 5 Chinese names: {category_lines['Chinese'][:5]}")
    
    n_hidden = 128
    n_epochs = 100000
    print_every = 5000
    plot_every = 1000
    learning_rate = 0.005

    model = RNN(n_letters, n_hidden, n_categories)
    print(model)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss() # negative log likelihood loss
    TextLearner.train(model, n_epochs, print_every, plot_every, learning_rate)

    model = torch.load('char_rnn_names.pt')
    TextLearner.predict(model, 'Wu')
    TextLearner.predict(model, 'Harry')
    TextLearner.predict(model, 'Louis')
    
    TextLearner.eval_all(model)
