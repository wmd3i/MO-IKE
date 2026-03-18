import torch
import random
import json
from tqdm import tqdm

from utils.llm_utils import load_model, generate_response, compute_reward
from retriever import Retriever
from utils.icl_utils import construct_icl_examples
from train import set_seed, stochastic

def eval(retriever, llm_pipeline, eval_data, corpus_idx, demos, lines):
    total_predictions = 0

    edit_incorrect_indices = []
    edit_correct_predictions = 0
    
    para_incorrect_indices = []
    para_correct_predictions = 0

    reten_incorrect_indices = []
    reten_correct_predictions = 0

    retriever.eval()

    for idx in tqdm(eval_data):
        icl_copy, icl_update, icl_retain = construct_icl_examples(idx, demos, order, corpus_idx)
        icl_examples = icl_copy + icl_update + icl_retain
        line = lines[idx]
        query = line['requested_rewrite']['prompt'].format(line['requested_rewrite']['subject'])
        _, probs, _ = retriever.retrieve_facts_list(query, icl_retain)
        print(probs)
        retrieved_examples = retriever.construct_retrieved_examples(query, icl_examples)
        print(len(retrieved_examples))

        prompt = line['requested_rewrite']['prompt']
        subject = line['requested_rewrite']['subject']
        prompt_calibrate = prompt.format('SUBJECT')
        prompt = prompt.format(subject)
        target_true = line['requested_rewrite']['target_true']['str']
        target_new = line['requested_rewrite']['target_new']['str']
        targets = [target_new, target_true]
        ground_truth = targets[0]  # or 1 if you're evaluating on true fact
        paraphrase = line['paraphrase_prompts'][0]
        neighbor= line['neighborhood_prompts'][0]
        retrieved_examples.append(f'New Fact: {prompt} {target_new}\nPrompt: {prompt} {target_new}\n\n')
        icl_edit, icl_para, icl_reten = [retrieved_examples.copy() for _ in range(3)]

        # Evaluate specific metric
        icl_edit.append(f'Prompt: {prompt}')
        icl_para.append(f'Prompt: {paraphrase}')
        icl_reten.append(f'Prompt: {neighbor}')

        edit_prompts = "".join(icl_edit)
        para_prompts = "".join(icl_para)
        reten_prompts = "".join(icl_reten)

        edit_answer = generate_response(edit_prompts, llm_pipeline)
        para_answer = generate_response(para_prompts, llm_pipeline)
        reten_answer = generate_response(reten_prompts, llm_pipeline)

        # Prompt into LLM and evaluate
        total_predictions += 1
        edit_reward = compute_reward(edit_answer.strip().lower(), ground_truth.strip().lower())
        para_reward = compute_reward(para_answer.strip().lower(), ground_truth.strip().lower())
        reten_reward = compute_reward(reten_answer.strip().lower(), target_true.strip().lower())

        if edit_reward:
            edit_correct_predictions += 1
        else:
            edit_incorrect_indices.append(idx)
        
        if para_reward:
            para_correct_predictions += 1
        else:
            para_incorrect_indices.append(idx) 
        
        if reten_reward:
            reten_correct_predictions += 1
        else:
            reten_incorrect_indices.append(idx)

    # Print final edit success
    edit_accuracy = (edit_correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0.0
    print(f"\nOverall Edit Success Rate: {edit_accuracy:.2f}% ({edit_correct_predictions}/{total_predictions})")
    print(f"Indices of incorrect edits in success rate: {edit_incorrect_indices}")

    para_accuracy = (para_correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0.0
    print(f"\nOverall Paraphrase Consistency Rate: {para_accuracy:.2f}% ({para_correct_predictions}/{total_predictions})")
    print(f"Indices of incorrect edits in success rate: {para_incorrect_indices}")

    reten_accuracy = (reten_correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0.0
    print(f"\nOverall Retention Rate: {reten_accuracy:.2f}% ({reten_correct_predictions}/{total_predictions})")
    print(f"Indices of incorrect edits in success rate: {reten_incorrect_indices}")


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    id = '' # Please use your huggingface ID to access the LLM
    llm_pipeline = load_model(id, model_id='meta-llama/Llama-3.2-3B-Instruct', device=device)
    
    retriever = Retriever()
    retriever = retriever.to(device)
    retriever.eval()
    
    # Default dataset (You may change this part to your own configuration)
    lines = []
    with open('./Datasets/counterfact.json', 'r') as f:
        lines = json.load(f)

    indices = []
    with open('./Datasets/corpus_idx.txt', 'r') as fIn:
        indices = fIn.readlines()
        indices = [index[:-1] for index in indices]

        corpus_idx = [[int(i) for i in index.split()] for index in indices]

    # Default evaluation data (You may change this part to your own configuration)
    num_samples = 600
    set_seed(42)
    data = [random.randint(0,1999) for _ in range(num_samples)]
    stochastic()
    eval_data = data[300:]
    
    demos = lines[2000:]
    lines = lines[:2000]
    order = [0]*4 + [1]*12 +[2]*16

    # Compute final editing reliability, generality, and specificity
    eval(retriever, llm_pipeline, eval_data, corpus_idx, demos, lines)
