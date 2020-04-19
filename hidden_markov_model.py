from nltk import trigrams
from markov_model import MarkovBase

import logging
import random
import re

class HiddenMarkovModel(MarkovBase):
    def __init__(self, vocab, random_seed):
        super().__init__(vocab, random_seed)

        self.order = 1

        # construct `counts`: the dimension should be 4x4
        self.counts = {}
        for char_first in self.vocab:
            self.counts[char_first] = {}
            for char_target in self.vocab:
                self.counts[char_first][char_target] = 0

        # construct {{current: next}: count} state count dictionary
        self.state_high = {}
        self.state_low = {}
        self.state_count = {}
        self.state_change = {
            "h2h": 0,
            "h2l": 0,
            "l2l": 0,
            "l2h": 0,
        }
        self.state_change_prob = {
            "h2h": 0,
            "h2l": 0,
            "l2l": 0,
            "l2h": 0,
        }

        # decide the prob of deciding the first char of sequences: equal distribution
        self.first_choice_prob = {}
        for char in self.vocab:
            self.first_choice_prob[char] = 1/(self.vocab_size)

        self.cond_prob = None

    def _to_cond_prob(self):
        '''
        Convert the number of counts in `self.cond_prob` into probabilities which should sum to 1.
        '''
        self.cond_prob = self.counts.copy()
        for char in self.counts.keys():
            occur_count = 0
            for target in self.counts[char].keys():
                occur_count += self.counts[char][target]
            if occur_count == 0:
                self.cond_prob[char][target] = 0
            else:
                for target in self.counts[char].keys():
                    self.cond_prob[char][target] = self.counts[char][target]/occur_count
        return

    def get_state_prob(self):
        '''
        Get the high probability state and low probability state
        '''
        for first, v in self.cond_prob.items():
            self.state_high[first] = {}
            self.state_low[first] = {}
            for second, prob in v.items():
                if prob >= 0.25:
                    self.state_high[first][second] = prob
                elif prob < 0.25:
                    self.state_low[first][second] = prob
        return

    def get_state_change(self, seq):
        '''
        get the trigram of sequences to calculate the   
        '''
        seq_trigrams = list(trigrams(seq))

        # construct `counts`: the dimension should be 4x4x4
        base_permutations = []
        for char_first in self.vocab:
            for char_second in self.vocab:
                for char_target in self.vocab:
                    base_permutations.append((char_first, char_second, char_target))

        for p in base_permutations:
            if p not in seq_trigrams:
                self.state_count[p] = 0
                continue
            for s in seq_trigrams:
                if p == s and p not in self.state_count.keys():
                    self.state_count[p] = 1
                elif p == s and p in self.state_count.keys():
                    self.state_count[p] += 1
        
        for t, c in self.state_count.items():
            if c > 0:
                if t[1] in self.state_high[t[0]].keys() and t[2] in self.state_high[t[1]].keys():
                    self.state_change["h2h"] += c
                elif t[1] in self.state_high[t[0]].keys() and t[2] in self.state_low[t[1]].keys():
                    self.state_change["h2l"] += c
                elif t[1] in self.state_low[t[0]].keys() and t[2] in self.state_high[t[1]].keys():
                    self.state_change["l2h"] += c
                elif t[1] in self.state_low[t[0]].keys() and t[2] in self.state_low[t[1]].keys():
                    self.state_change["l2l"] += c
                else:
                    print("Something is damn wrong")
        
        for t, c in self.state_change.items():
            print(f"{t}: {c}")
            self.state_change_prob[t] = c/sum(self.state_change.values())
        return

    def fit(self, seq):
        '''
        Calculate subsequence occurrences and convert it into conditional probabilities.
        '''
        if not isinstance(seq, str):
            raise TypeError('Invalid parameter `seq`.')

        if len(seq) < self.order+1:
            raise ValueError('Invalid parameter `seq`.')

        seq_len = len(seq)
        pair_num = seq_len - self.order

        for index, char in enumerate(seq):
            if index < pair_num:
                self.counts[char][seq[index+1]] += 1
        
        logging.info(f'Counts: {self.counts}')
        self._to_cond_prob()
        logging.info(f'Cond_prob: {self.cond_prob}')
        self.get_state_prob()
        logging.info(f'State_High: {self.state_high}')
        logging.info(f'State_Low: {self.state_low}')
        self.get_state_change(seq)
        logging.info(f'State_Count: {self.state_count}')
        logging.info(f'State_Change: {self.state_change}')
        logging.info(f'State_Change_Prob: {self.state_change_prob}')
        return

def test():
    logging.basicConfig(level=logging.INFO,
            format='\n%(asctime)s %(name)-5s === %(levelname)-5s === \n%(message)s\n')

    seq = "atccatgcatgaccatggtcag"
    print(f'Target sequence: {seq}')

    # test hidden markov model
    print('\n=== Hidden Markov Model ===')
    hidden_markov_model = HiddenMarkovModel(vocab=set(seq), random_seed=17)
    hidden_markov_model.fit(seq)
    # generated_seq = hidden_markov_model.generate(len(seq))
    # print(f'Generated sequence: {generated_seq}')
    # print(f'Target sequence generation probability: {hidden_markov_model.generating_prob(seq)}')

if __name__=="__main__":
    test()
