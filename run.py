import logging
from argparse import ArgumentParser
import timeit

from markov_model import MarkovOrderZero, MarkovOrderOne, MarkovOrderTwo
from hidden_markov_model import HiddenMarkovModel

class RecordTime(object):
    def __init__(self):
        super().__init__()
    
    def start(self):
        self.start_time = timeit.default_timer()

    def stop(self):
        self.end_time = timeit.default_timer()

    def get_duration(self):
        return self.end_time - self.start_time


def main():
    parser = ArgumentParser()
    parser.add_argument('-p', '--print-detail', help='Whether to print details.', action='store_true')
    args = parser.parse_args()

    timer = RecordTime()

    if args.print_detail:
        logging.basicConfig(level=logging.INFO,
                format='\n%(asctime)s %(name)-5s === %(levelname)-5s === %(message)s\n')

    # read target sequence
    seq_file_path = "./NC_000006_12_Homo_sapiens_chromosome_6_GRCh38_p13_Primary_Assembly.txt"
    with open(seq_file_path, 'r') as f:
        s = f.readline()
        s = s.strip('\n')
        s = s.lower()
    
    assert len(s) == 100000

    # read test sequence
    test_seq_file_path = "./NC_000006_12_Homo_sapiens_chromosome_6_GRCh38_p13_Primary_Assembly_test.txt"
    with open(test_seq_file_path, 'r') as f:
        test_s = f.readline()
        test_s = test_s.strip('\n')
        test_s = test_s.lower()
    
    assert len(test_s) == 100000

    # run Markov models
    model_infos = {
        'Markov Model Order 0':{
            'class': MarkovOrderZero,
        },
        'Markov Model Order 1':{
            'class': MarkovOrderOne,
        },
        'Markov Model Order 2':{
            'class': MarkovOrderTwo,
        },
    }

    for model_name in model_infos.keys():
        print(f'\n=== {model_name} ===')
        model = model_infos[model_name]['class'](vocab=set(s), random_seed=17)
        timer.start()
        model.fit(s)
        timer.stop()
        fit_time = timer.get_duration()
        timer.start()
        print(f'Target sequence generation probability:\t{model.generating_prob(s)}')
        timer.stop()
        calculate_gp_time = timer.get_duration()
        timer.start()
        print(f'Another 100k sequence generation probability:\t{model.generating_prob(test_s)}')
        timer.stop()
        calculate_another_gp_time = timer.get_duration()
        print(f'Time - fitting model:\t{fit_time:.3f} sec')
        print(f'Time - calculating target sequence generation probability:\t{calculate_gp_time:.3f} sec')
        print(f'Time - calculating another 100k sequence generation probability:\t{calculate_another_gp_time:.3f} sec')

    # run hidden Markov models
    print(f'\n=== Hidden Markov Model ===')
    hidden_markov_model = HiddenMarkovModel(vocab=set(s), random_seed=17)
    timer.start()
    hidden_markov_model.fit(s)
    timer.stop()
    fit_time = timer.get_duration()
    timer.start()
    print(f'Target sequence generation probability:\t{hidden_markov_model.generating_prob(s)}')
    timer.stop()
    calculate_gp_time = timer.get_duration()
    timer.start()
    print(f'Another 100k sequence generation probability:\t{hidden_markov_model.generating_prob(test_s)}')
    timer.stop()
    calculate_another_gp_time = timer.get_duration()
    print(f'Time - fitting model:\t{fit_time:.3f} sec')
    print(f'Time - calculating target sequence generation probability:\t{calculate_gp_time:.3f} sec')
    print(f'Time - calculating another 100k sequence generation probability:\t{calculate_another_gp_time:.3f} sec')

    # write state sequences to text files
    logging.info('Writing state sequences to file...')
    with open('./state_seq_s.txt', 'w') as f:
        f.write(''.join(hidden_markov_model.state_sequence(s)))
    with open('./state_seq_test_s.txt', 'w') as f:
        f.write(''.join(hidden_markov_model.state_sequence(test_s)))

    return

if __name__ == '__main__':
    main()
