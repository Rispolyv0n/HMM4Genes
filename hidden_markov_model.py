from nltk import trigrams
from markov_model import MarkovBase
import numpy as np

import logging
import random
import re
import math

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
        self.init_state_prob = {"h": 0, "l": 0}
        self.state_high = {}
        self.state_low = {}
        self.state_prob = {}
        self.state_count = {}
        self.state_change = {
            "h": {"h": 0, "l": 0},
            "l": {"h": 0, "l": 0},
        }
        self.state_change_prob = {
            "h": {"h": 0, "l": 0},
            "l": {"h": 0, "l": 0},
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
                if prob > 0.25:
                    self.state_high[first][second] = prob
                elif prob <= 0.25:
                    self.state_low[first][second] = prob

        for k, v in self.state_high.items():
            for _, prob in v.items():
                self.init_state_prob["h"] += prob

        for k, v in self.state_low.items():
            for _, prob in v.items():
                self.init_state_prob["l"] += prob

        self.init_state_prob["h"] = self.init_state_prob["h"]/4
        self.init_state_prob["l"] = self.init_state_prob["l"]/4

        total_prob = 0
        state_high_prob = {"a": 0, "c": 0, "g": 0, "t": 0}
        state_low_prob = {"a": 0, "c": 0, "g": 0, "t": 0}

        for key in state_high_prob.keys():
            for k, v in self.state_high.items():
                for element, prob in v.items():
                    if key == element:
                        state_high_prob[key] += prob
                        total_prob += prob
        
        for key in state_high_prob.keys():
            state_high_prob[key] = state_high_prob[key]/total_prob

        total_prob = 0

        for key in state_low_prob.keys():
            for k, v in self.state_low.items():
                for element, prob in v.items():
                    if key == element:
                        state_low_prob[key] += prob
                        total_prob += prob
        
        for key in state_low_prob.keys():
            state_low_prob[key] = state_low_prob[key]/total_prob

        self.state_prob["h"] = state_high_prob
        self.state_prob["l"] = state_low_prob
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
        
        for trigram, count in self.state_count.items():
            if count > 0:
                if trigram[1] in self.state_high[trigram[0]].keys() and trigram[2] in self.state_high[trigram[1]].keys():
                    self.state_change["h"]["h"] += count
                elif trigram[1] in self.state_high[trigram[0]].keys() and trigram[2] in self.state_low[trigram[1]].keys():
                    self.state_change["h"]["l"] += count
                elif trigram[1] in self.state_low[trigram[0]].keys() and trigram[2] in self.state_high[trigram[1]].keys():
                    self.state_change["l"]["h"] += count
                elif trigram[1] in self.state_low[trigram[0]].keys() and trigram[2] in self.state_low[trigram[1]].keys():
                    self.state_change["l"]["l"] += count

        for s_from, v in self.state_change.items():
            for s_to, count in v.items():
                self.state_change_prob[s_from][s_to] = count/(sum([c for k, c in v.items()]))
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
        
        # logging.info(f'Counts: {self.counts}')
        self._to_cond_prob()
        # logging.info(f'Cond_prob: {self.cond_prob}')
        self.get_state_prob()
        # logging.info(f'State_High: {self.state_high}')
        # logging.info(f'State_Low: {self.state_low}')
        logging.info(f'State_Prob: {self.state_prob}')
        logging.info(f'Init_State_Prob: {self.init_state_prob}')
        self.get_state_change(seq)
        # logging.info(f'State_Count: {self.state_count}')
        # logging.info(f'State_Change: {self.state_change}')
        logging.info(f'State_Change_Prob: {self.state_change_prob}')
        return

    def generating_prob(self, seq):
        cur_state_prob = np.array(list(v for k, v in self.init_state_prob.items()))
        state_change_prob = np.array(
            [
                [ self.state_change_prob[state_start][state_end] for state_end in ['h', 'l'] ]
                    for state_start in ['h', 'l']
            ]
        )
        output_prob = np.array(
            [ 
                [ self.state_prob[state][char] for char in self.vocab ] 
                    for state in ['h', 'l']
            ]
        )
        cur_prob = 0
        for char in seq:
            cur_state_prob = np.dot(cur_state_prob, state_change_prob)
            cur_output_prob = np.dot(cur_state_prob, output_prob)
            cur_prob += math.log(cur_output_prob[self.vocab2id[char]], 2)
        return cur_prob

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
    print(f'Target sequence generation probability: {hidden_markov_model.generating_prob(seq)}')

if __name__=="__main__":
    test()
