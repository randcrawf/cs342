from numpy import inf
from .models import LanguageModel, AdjacentLanguageModel, Bigram, load_model
from . import utils
import torch

def log_likelihood(model: LanguageModel, some_text: str):
    """
    Your code here

    Evaluate the log-likelihood of a given string.

    Hint: utils.one_hot might come in handy

    :param model: A LanguageModel
    :param some_text:
    :return: float
    """
    res = 0.
    preds = model.predict_all(some_text)
    oh = utils.one_hot(some_text)
    for i in range(oh.shape[0]):
        for j in range(oh.shape[1]):
            if(oh[i][j] == 1):
                res += preds[i][j]

    return res


def sample_random(model: LanguageModel, max_length: int = 100):
    """
    Your code here.

    Sample a random sentence from the language model.
    Terminate once you reach a period '.'

    :param model: A LanguageModel
    :param max_length: The maximum sentence length
    :return: A string
    """
    text = ""
    while(len(text) < max_length):
        index = 0
        pred = model.predict_next(text)

        index = torch.distributions.Categorical(logits=pred).sample()

        text += utils.vocab[index]
        if text[len(text) - 1] == '.':
            break
    return text


class TopNHeap:
    """
    A heap that keeps the top N elements around
    h = TopNHeap(2)
    h.add(1)
    h.add(2)
    h.add(3)
    h.add(0)
    print(h.elements)
    > [2,3]

    """
    def __init__(self, N):
        self.elements = []
        self.N = N

    def add(self, val, s):
        from heapq import heappush, heapreplace
        if len(self.elements) < self.N:
            heappush(self.elements, (val, s))
        elif self.elements[0] < (val, s):
            heapreplace(self.elements, (val, s))


def beam_search(model: LanguageModel, beam_size: int, n_results: int = 10, max_length: int = 100, average_log_likelihood: bool = False):
    """
    Your code here

    Use beam search for find the highest likelihood generations, such that:
      * No two returned sentences are the same
      * the `log_likelihood` of each returned sentence is as large as possible

    :param model: A LanguageModel
    :param beam_size: The size of the beam in beam search (number of sentences to keep around)
    :param n_results: The number of results to return
    :param max_length: The maximum sentence length
    :param average_log_likelihood: Pick the best beams according to the average log-likelihood, not the sum
                                   This option favors longer strings.
    :return: A list of strings of size n_results
    """
    
    topN = TopNHeap(beam_size)
    complete_sentences = TopNHeap(n_results)
    for j in range(len(utils.vocab)):
        c = utils.vocab[j]
        if c == ".":
            complete_sentences.add(log_likelihood(model, c), c)
        else:
            topN.add(log_likelihood(model, c), c)

    count = 1
    while count < max_length - 1:
        print(count)
        prev_beam = topN.elements.copy()
        topN = TopNHeap(beam_size)
        for i in range(len(prev_beam)):
            val, s = prev_beam[i]
            for j in range(len(utils.vocab)):
                c = utils.vocab[j]
                ll = log_likelihood(model, s + c)
                if c == ".":
                    complete_sentences.add(ll / (len(s) + 1 if average_log_likelihood else 1), s + c)
                else:
                    topN.add(ll, s + c)
        count += 1

    for _val, s in topN.elements:
        ll = log_likelihood(model, s + ".")
        complete_sentences.add(ll / (len(s) + 1 if average_log_likelihood else 1), s + ".")
    
    res = ["" for _ in range(n_results)]
    for i in range(len(complete_sentences.elements)):
        _val, s = complete_sentences.elements[i]
        res[i] = s
    print(complete_sentences.elements)

    return res



if __name__ == "__main__":
    """
      Some test code.
    """
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-m', '--model', choices=['Adjacent', 'Bigram', 'TCN'], default='Adjacent')
    args = parser.parse_args()

    lm = AdjacentLanguageModel() if args.model == 'Adjacent' else (load_model() if args.model == 'TCN' else Bigram())

    for s in ['abcdefg', 'abcgdef', 'abcbabc', '.abcdef', 'fedcba.']:
        print(s, float(log_likelihood(lm, s)))
    print()

    for i in range(10):
        s = sample_random(lm)
        print(s, float(log_likelihood(lm, s)) / len(s))
    print()

    for s in beam_search(lm, 100):
        print(s, float(log_likelihood(lm, s)) / len(s))
    print()

    for s in beam_search(lm, 100, average_log_likelihood=True):
        print(s, float(log_likelihood(lm, s)) / len(s))
