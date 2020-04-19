import logging
from argparse import ArgumentParser
import sys

from markov_model import MarkovOrderZero, MarkovOrderOne, MarkovOrderTwo

def main():
    parser = ArgumentParser()
    parser.add_argument('-p', '--print-detail', help='Whether to print details.', action='store_true')
    args = parser.parse_args()

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
        model.fit(s)
        print(f'Target sequence generation probability: {model.generating_prob(s)}')
        print(f'Another 100k sequence generation probability: {model.generating_prob(test_s)}')

    # TODO: run hidden Markov models
    
    return

if __name__ == '__main__':
    main()
