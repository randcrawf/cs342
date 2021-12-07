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
    likelihood = 0.0
    x = model.predict_all(some_text)
    r = utils.one_hot(some_text)
    for i in range(r.shape[0]):
        for j in range(r.shape[1]):
            item = r[i][j]
            if(item == 1):
                likelihood += x[i][j]

    return likelihood


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
        x = model.predict_all(text)[:,-1]

        index = torch.distributions.Categorical(logits=x).sample()

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
        # print(val, s)
        # print(self.elements)
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

    all_found = False
    topN = TopNHeap(beam_size)
    if average_log_likelihood:
        # c = 0
        # while c > max_length and not all_found:
        #     print(utils.vocab)
        #     all_found = True
        #     for i in range(len(topN.elements)):
        #         val, s = topN.elements[i]
        #         if s[-1] != '.':
        #             all_found = False
        #             preds = model.predict_all(s)
                    
        #             for j in range(len(preds)):
        #                 c = chr(j + ord('a'))
        #                 if j == len(utils.vocab) - 1:
        #                     c = "."
        #                 elif j == len(utils.vocab) - 2:
        #                     c = " "
        #                 topN.add(val + preds[j], s + c)

        #     c += 1
        return [("" + str(i)) for i in range(n_results)]
    else: 
        count = 0
        while count < max_length and not all_found:
            print(count)
            prev_beam = topN.elements.copy()
            topN = TopNHeap(beam_size)
            all_found = True
            for i in range(1 if len(prev_beam) == 0 else len(prev_beam)):
                if len(prev_beam) > 0:
                    val, s = prev_beam[i]
                    if s[-1] == '.':
                        topN.add(val, s)
                    else:
                        all_found = False
                        for j in range(len(utils.vocab)):
                            c = chr(j + ord('a'))
                            if j == len(utils.vocab) - 1:
                                c = "."
                            elif j == len(utils.vocab) - 2:
                                c = " "
                            topN.add(log_likelihood(model, s + c), s + c)
                        # print("x4")
                else:
                    all_found = False
                    # print("x5")
                    for j in range(len(utils.vocab)):
                        c = chr(j + ord('a'))
                        if j == len(utils.vocab) - 1:
                            c = "."
                        elif j == len(utils.vocab) - 2:
                            c = " "
                        topN.add(log_likelihood(model, c), c)
                    # print("x6")
            # print("x7")
            count += 1
    # print("x8")
    results = TopNHeap(n_results)
    for val, s in topN.elements:
        results.add(val, s)
    # print(topN.elements)
    res = ["" for _ in range(n_results)]
    for i in range(len(results.elements)):
        val, s = results.elements[i]
        res[i] = s
    print(results.elements)

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
