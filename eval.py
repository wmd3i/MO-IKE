from retriever import Retriever, sample_retrieved_examples
from utils.icl_utils import construct_icl_examples
import json

if __name__ == "__main__":
    lines = []
    with open('./Datasets/counterfact.json', 'r') as f:
        lines = json.load(f)

    indices = []
    with open('./Datasets/corpus_idx.txt', 'r') as fIn:
        indices = fIn.readlines()
        indices = [index[:-1] for index in indices]

        corpus_idx = [[int(i) for i in index.split()] for index in indices]
    
    demos = lines[2000:]
    lines = lines[:2000]
    order = [0]*4 + [1]*12 +[2]*16

    idx = 0
    icl_copy, icl_update, icl_retain = construct_icl_examples(idx, demos, order, corpus_idx)
    icl_examples = icl_copy + icl_update + icl_retain

    retriever = Retriever()
    line = lines[idx]
    query = line['requested_rewrite']['prompt'].format(line['requested_rewrite']['subject'])
    retrieved_examples = retriever.construct_retrieved_examples(query, icl_examples)
    print(f"Sanity check: the current of number of examples is: {len(retrieved_examples)}")
    