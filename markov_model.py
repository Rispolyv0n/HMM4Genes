#!/usr/bin/python
# -*- coding: utf-8 -*-

# Import required modules
import random
import logging
import math


class MarkovBase(object):
    def __init__(self, vocab, random_seed):
        super().__init__()
        
        if not isinstance(vocab, list) and not isinstance(vocab, set):
            raise TypeError('Invalid parameter `vocab`.')

        if not isinstance(random_seed, int):
            raise TypeError('Invalid parameter `random_seed`.')

        self.order = None

        self.random_seed = random_seed
        random.seed(self.random_seed)

        self.vocab = sorted(vocab)
        self.vocab_size = len(self.vocab)

        # construct `vocab2id` & `id2vocab`        
        self.vocab2id = {}
        self.id2vocab = {}
        for index, char in enumerate(self.vocab):
            self.vocab2id[char] = index
            self.id2vocab[index] = char

        # decide the prob of deciding the first char of sequences: equal distribution
        self.first_choice_prob = {}
        for char in self.vocab:
            self.first_choice_prob[char] = 1/(self.vocab_size)

        self.cond_prob = None
        return
    
    
    def generate(self, seq_len):
        '''
        Generate a sequence with the fitted conditional probability.
        '''
        if not isinstance(seq_len, int):
            raise TypeError('Invalid parameter `seq_len`.')

        if seq_len < 1:
            raise ValueError('Invalid parameter `seq_len`.')
        
        seq = ""
        for i in range(self.order):
            seq += self._first_choice()
        for i in range(seq_len-self.order):
            next_char = self._next_choice(seq[-self.order:])
            if next_char == None:
                return seq
            else:
                seq += self._next_choice(seq[-self.order:])

        return seq
    
    def _first_choice(self):
        '''
        Generate the first character of the sequence.
        The probabilities of deciding which character is going to be generated are equal.
        '''
        random_num = random.random()
        for index in range(self.vocab_size):
            cur_threshold = (index+1) * (1/self.vocab_size)
            if random_num <= cur_threshold:
                return self.id2vocab[index]


class MarkovOrderZero(object):
    def __init__(self, vocab, random_seed):
        super().__init__()
        
        if not isinstance(vocab, list) and not isinstance(vocab, set):
            raise TypeError('Invalid parameter `vocab`.')

        if not isinstance(random_seed, int):
            raise TypeError('Invalid parameter `random_seed`.')

        self.order = 0

        self.random_seed = random_seed
        random.seed(self.random_seed)

        self.vocab = sorted(vocab)
        self.vocab_size = len(self.vocab)

        # construct `vocab2id` & `id2vocab`        
        self.vocab2id = {}
        self.id2vocab = {}
        for index, char in enumerate(self.vocab):
            self.vocab2id[char] = index
            self.id2vocab[index] = char

        # construct `counts`: the dimension should be 4
        self.counts = {}
        for char in self.vocab:
            self.counts[char] = 0

        self.cond_prob = None
        return


    def fit(self, seq):
        '''
        Calculate subsequence occurrences and convert it into conditional probabilities.
        '''
        if not isinstance(seq, str):
            raise TypeError('Invalid parameter `seq`.')

        if len(seq) < self.order+1:
            raise ValueError('Invalid parameter `seq`.')

        for char in seq:
            self.counts[char] += 1

        logging.info(f'Counts: {self.counts}')
        self._adjust_cond_prob()
        logging.info(f'Cond_prob: {self.cond_prob}')
        return
    
    def _adjust_cond_prob(self):
        '''
        Convert the number of counts in `self.cond_prob` into probabilities which should sum to 1.
        '''
        self.cond_prob = self.counts.copy()
        occur_count = 0
        for char in self.counts.keys():
            occur_count += self.counts[char]
        for char in self.counts.keys():
            if occur_count == 0:
                self.cond_prob[char] = 0
            else:
                self.cond_prob[char] = self.counts[char]/occur_count

        return


    def generate(self, seq_len):
        '''
        Generate a sequence with the fitted conditional probability.
        '''
        if not isinstance(seq_len, int):
            raise TypeError('Invalid parameter `seq_len`.')

        if seq_len < 1:
            raise ValueError('Invalid parameter `seq_len`.')
        
        seq = ""
        for i in range(seq_len):
            next_char = self._next_choice()
            if next_char == None:
                return seq
            else:
                seq += self._next_choice()

        return seq

    def _next_choice(self):
        '''
        Generate the next character.
        The probabilities of deciding which character is going to be generated are based on `self.cond_prob`.
        '''
        if self.cond_prob == None:
            raise UnboundLocalError('Model hasn\'t been fitted yet.')
        
        if sum(prob for char, prob in self.cond_prob.items()) == 0:
            return None

        random_num = random.random()
        
        cur_threshold = 0
        for index, (char, prob) in enumerate(self.cond_prob.items()):
            if prob > 0:
                cur_threshold += prob
                if random_num <= cur_threshold:
                    return char

        return self.cond_prob.items()[-1][1]

    def generating_prob(self, seq):
        '''
        Calculate the (log base 2) probabilitiy of generating a given sequence.
        '''
        if self.cond_prob == None:
            raise UnboundLocalError('Model hasn\'t been fitted yet.')

        if not isinstance(seq, str):
            raise TypeError('Invalid parameter `seq`.')

        log_base = 2
        prob = 0
        for index, char in enumerate(seq):
            if self.cond_prob[char] == 0:
                return 0
            else:
                prob += math.log(self.cond_prob[char], log_base)

        return prob


class MarkovOrderOne(MarkovBase):
    def __init__(self, vocab, random_seed):
        super().__init__(vocab, random_seed)

        self.order = 1

        # construct `counts`: the dimension should be 4x4
        self.counts = {}
        for char_first in self.vocab:
            self.counts[char_first] = {}
            for char_target in self.vocab:
                self.counts[char_first][char_target] = 0

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
        self._adjust_cond_prob()
        logging.info(f'Cond_prob: {self.cond_prob}')

        return

    def _adjust_cond_prob(self):
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

    def _next_choice(self, cur_seq):
        '''
        Generate the next character based on the given sequence.
        The probabilities of deciding which character is going to be generated are based on `self.cond_prob`.
        '''
        if self.cond_prob == None:
            raise UnboundLocalError('Model hasn\'t been fitted yet.')

        if not isinstance(cur_seq, str):
            raise TypeError(f'Invalid parameter `cur_seq`: {cur_seq}')

        if len(cur_seq) < self.order:
            raise ValueError(f'Invalid parameter `cur_seq`: {cur_seq}')
        
        random_num = random.random()
        sorted_choices = {k: v for k, v in sorted(self.cond_prob[cur_seq].items(), key=lambda item: item[1])}

        cur_threshold = 0
        for index, (char, prob) in enumerate(sorted_choices.items()):
            if prob > 0:
                cur_threshold += prob
                if random_num <= cur_threshold:
                    return char

        if sum(prob for char, prob in sorted_choices.items()) == 0:
            return None
        else:
            return sorted_choices.items()[-1][1]

    def generating_prob(self, seq):
        '''
        Calculate the (log base 2) probabilitiy of generating a given sequence.
        '''
        if self.cond_prob == None:
            raise UnboundLocalError('Model hasn\'t been fitted yet.')

        if not isinstance(seq, str):
            raise TypeError('Invalid parameter `seq`.')

        log_base = 2
        prob = 0
        for i in range(self.order):
            prob += math.log(self.first_choice_prob[seq[i]], log_base)

        for index, char in enumerate(seq[self.order:]):
            if self.cond_prob[seq[index]][char] == 0:
                return 0
            else:
                prob += math.log(self.cond_prob[seq[index]][char], log_base)

        return prob


class MarkovOrderTwo(MarkovBase):
    def __init__(self, vocab, random_seed):
        super().__init__(vocab, random_seed)

        self.order = 2

        # construct `counts`: the dimension should be 4x4x4
        self.counts = {}
        for char_first in self.vocab:
            self.counts[char_first] = {}
            for char_second in self.vocab:
                self.counts[char_first][char_second] = {}
                for char_target in self.vocab:
                    self.counts[char_first][char_second][char_target] = 0

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
                self.counts[char][seq[index+1]][seq[index+2]] += 1
        
        logging.info(f'Counts: {self.counts}')
        self._adjust_cond_prob()
        logging.info(f'Cond_prob: {self.cond_prob}')
        return

    def _adjust_cond_prob(self):
        '''
        Convert the number of counts in `self.cond_prob` into probabilities which should sum to 1.
        '''
        self.cond_prob = self.counts.copy()
        for char_first in self.counts.keys():
            for char_second in self.counts[char_first].keys():
                occur_count = 0
                for target in self.counts[char_first][char_second].keys():
                    occur_count += self.counts[char_first][char_second][target]
                if occur_count == 0:
                    self.cond_prob[char_second][char_second][target] = 0
                else:
                    for target in self.counts[char_first][char_second].keys():
                        self.cond_prob[char_first][char_second][target] = self.counts[char_first][char_second][target]/occur_count
        
        return

    def _next_choice(self, cur_seq):
        '''
        Generate the next character based on the given sequence.
        The probabilities of deciding which character is going to be generated are based on `self.cond_prob`.
        '''
        if self.cond_prob == None:
            raise UnboundLocalError('Model hasn\'t been fitted yet.')

        if not isinstance(cur_seq, str):
            raise TypeError(f'Invalid parameter `cur_seq`: {cur_seq}')

        if len(cur_seq) < self.order:
            raise ValueError(f'Invalid parameter `cur_seq`: {cur_seq}')
        
        random_num = random.random()
        sorted_choices = {k: v for k, v in sorted(self.cond_prob[cur_seq[0]][cur_seq[1]].items(), key=lambda item: item[1])}

        cur_threshold = 0
        for index, (char, prob) in enumerate(sorted_choices.items()):
            if prob > 0:
                cur_threshold += prob
                if random_num <= cur_threshold:
                    return char
        
        if sum(prob for char, prob in sorted_choices.items()) == 0:
            return None
        else:
            return sorted_choices.items()[-1][1]
        
    def generating_prob(self, seq):
        '''
        Calculate the (log base 2) probabilitiy of generating a given sequence.
        '''
        if self.cond_prob == None:
            raise UnboundLocalError('Model hasn\'t been fitted yet.')
        
        if not isinstance(seq, str):
            raise TypeError('Invalid parameter `seq`.')

        log_base = 2
        prob = 0
        for i in range(self.order):
            prob += math.log(self.first_choice_prob[seq[i]], log_base)

        for index, char in enumerate(seq[self.order:]):
            if self.cond_prob[seq[index]][seq[index+1]][char] == 0:
                return 0
            else:
                prob += math.log(self.cond_prob[seq[index]][seq[index+1]][char], log_base)

        return prob



def test():
    logging.basicConfig(level=logging.INFO,
            format='\n%(asctime)s %(name)-5s === %(levelname)-5s === %(message)s\n')

    seq = "atccatgcatgcag"
    print(f'Target sequence: {seq}')

    # test markov model order 0
    print('\n=== Markov Model Order 0 ===')
    markov_model_zero = MarkovOrderZero(vocab=set(seq), random_seed=17)
    markov_model_zero.fit(seq)
    generated_seq = markov_model_zero.generate(len(seq))
    print(f'Generated sequence: {generated_seq}')
    print(f'Target sequence generation probability: {markov_model_zero.generating_prob(seq)}')

    # test markov model order 1
    print('\n=== Markov Model Order 1 ===')
    markov_model_one = MarkovOrderOne(vocab=set(seq), random_seed=17)
    markov_model_one.fit(seq)
    generated_seq = markov_model_one.generate(len(seq))
    print(f'Generated sequence: {generated_seq}')
    print(f'Target sequence generation probability: {markov_model_one.generating_prob(seq)}')
    
    # test markov model order 2
    print('\n=== Markov Model Order 2 ===')
    markov_model_two = MarkovOrderTwo(vocab=set(seq), random_seed=17)
    markov_model_two.fit(seq)
    generated_seq = markov_model_two.generate(len(seq))
    print(f'Generated sequence: {generated_seq}')
    print(f'Target sequence generation probability: {markov_model_two.generating_prob(seq)}')

    return


if __name__ == '__main__':
    test()
