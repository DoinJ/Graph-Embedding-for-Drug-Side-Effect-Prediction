#!/usr/bin/env python
# coding: utf-8

# In[1]:


import datetime
import getpass
import json
import os
import random
import time
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import Ipynb_importer
import numpy as np

from embed_train import embedding_training, load_embedding, read_node_labels, split_train_test_graph
from evaluation import LinkPrediction      


# In[3]:


# choose the type of model and perform the link prediction task
def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument('--input', required=True,
                        help='Input graph file. Only accepted edgelist format.')
    parser.add_argument('--output',
                        help='Output graph embedding file', required=True)
    parser.add_argument('--task', choices=[
        'none',
        'link-prediction'], default='none',
                        help='Choose to evaluate the embedding quality based on a specific prediction task. '
                             'None represents no evaluation, and only run for training embedding.')
    parser.add_argument('--testingratio', default=0.1, type=float,
                        help='Testing set ratio for prediction tasks.'
                             'In link prediction, it splits all the known edges. ')
    parser.add_argument('--number-walks', default=32, type=int,
                        help='Number of random walks to start at each node. '
                             'Only for random walk-based methods: DeepWalk, node2vec, struc2vec')
    parser.add_argument('--walk-length', default=64, type=int,
                        help='Length of the random walk started at each node. '
                             'Only for random walk-based methods: DeepWalk, node2vec, struc2vec')
    parser.add_argument('--workers', default=8, type=int,
                        help='Number of parallel processes. '
                             'Only for random walk-based methods: DeepWalk, node2vec, struc2vec')
    parser.add_argument('--dimensions', default=100, type=int,
                        help='the dimensions of embedding for each node.')
    parser.add_argument('--window-size', default=10, type=int,
                        help='Window size of word2vec model. '
                             'Only for random walk-based methods: DeepWalk, node2vec, struc2vec')
    parser.add_argument('--p', default=1.0, type=float,
                        help='p is a hyper-parameter for node2vec, '
                             'and it controls how fast the walk explores.')
    parser.add_argument('--q', default=1.0, type=float,
                        help='q is a hyper-parameter for node2vec, '
                             'and it controls how fast the walk leaves the neighborhood of starting node.')
    parser.add_argument('--method', required=True, choices=[
        'DeepWalk',
        'node2vec',
    ], help='The embedding learning method')

    parser.add_argument('--weighted', type=bool, default=False,
                        help='Treat graph as weighted')
    parser.add_argument('--directed', type=bool, default=False,
                        help='Treat graph as directed')
    parser.add_argument('--eval-result-file', help='save evaluation performance')
    parser.add_argument('--seed',default=0, type=int,  help='seed value')
    args = parser.parse_args(args=['--input','.../data/drug_combinations.txt',
                                   '--output','.../node2vec_embeddings3.txt',
                                   '--task','link-prediction',
                                   '--method','node2vec',
                                   '--eval-result-file','.../results.txt'])
    # replace "..." in the arguments with your own path of the drug_combination file and your expected output path
    return args  

def main(args):
    print('#' * 70)
    print('Embedding Method: %s, Evaluation Task: %s' % (args.method, args.task))
    print('#' * 70)

    if args.task == 'link-prediction':
        G, G_train, testing_pos_edges, train_graph_filename = split_train_test_graph(args.input, args.seed, weighted=args.weighted)
        time1 = time.time()
        embedding_training(args, train_graph_filename)
        embed_train_time = time.time() - time1
        print('Embedding Learning Time: %.2f s' % embed_train_time)
        embedding_look_up = load_embedding(args.output)
        time1 = time.time()
        print('Begin evaluation...')
        result = LinkPrediction(embedding_look_up, G, G_train, testing_pos_edges,args.seed)
        eval_time = time.time() - time1
        print('Prediction Task Time: %.2f s' % eval_time)
        os.remove(train_graph_filename)
    else:
        train_graph_filename = args.input
        time1 = time.time()
        embedding_training(args, train_graph_filename)
        embed_train_time = time.time() - time1
        print('Embedding Learning Time: %.2f s' % embed_train_time)

    if args.eval_result_file and result:
        _results = dict(
            input=args.input,
            task=args.task,
            method=args.method,
            dimension=args.dimensions,
            user=getpass.getuser(),
            date=datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S'),
            seed=args.seed,
        )

        if args.task == 'link-prediction':
            auc_roc, auc_pr, accuracy, f1 = result
            _results['results'] = dict(
                auc_roc=auc_roc,
                auc_pr=auc_pr,
                accuracy=accuracy,
                f1=f1,
            )
        else:
            accuracy, f1_micro, f1_macro = result
            _results['results'] = dict(
                accuracy=accuracy,
                f1_micro=f1_micro,
                f1_macro=f1_macro,
            )

        with open(args.eval_result_file, 'a+') as wf:
            print(json.dumps(_results, sort_keys=True), file=wf)


def more_main():
    args = parse_args()
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    main(parse_args())


if __name__ == "__main__":
    more_main() 

