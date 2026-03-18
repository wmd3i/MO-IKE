"""
Dynamic Retriever fromulated in MO-IKE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
from utils.icl_utils import construct_icl_examples
import json

# -----------------------------
# Define the Retriever Model
# -----------------------------
class Retriever(nn.Module):
    def __init__(self):
        super(Retriever, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        for param in self.bert.parameters():
            param.requires_grad = False
        H = self.bert.config.hidden_size
        self.query_actor = nn.Linear(H, H)
        self.cand_actor = nn.Linear(H, H)
        self.stop_embedding = nn.Parameter(torch.randn(H))
        self.stop_bias = nn.Parameter(torch.tensor(-10.0))
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


    def retrieve_facts_list(self, query, facts):
        """
        Retrieve closest matching facts from facts given a query.
        """
        query_output = self.__get_embeddings(query)
        query_vec = self.query_actor(query_output)
        query_vec = F.normalize(query_vec, p=2, dim=1)

        cand_output = self.__get_embeddings(facts)
        cand_vec = self.cand_actor(cand_output)
        cand_vec = F.normalize(cand_vec, p=2, dim=1)

        stop_vec = F.normalize(self.stop_embedding, p=2, dim=0)

        temperature = 1.0

        scores = torch.matmul(query_vec, cand_vec.t()).view(-1) * temperature
        stop_score = torch.matmul(query_vec, stop_vec.t()).view(1) * temperature + self.stop_bias

        all_scores = torch.cat((scores.view(-1), stop_score.view(-1)), dim = 0)

        probs = torch.softmax(all_scores, dim=0)

        prob_fact_pair = {facts[i]: probs[i].item() for i in range(len(facts))}
        prob_fact_pair["STOP"] = probs[-1].item() # Include STOP signal

        return all_scores, probs, prob_fact_pair

    def construct_retrieved_examples(self, query, facts):
        """
        Construct list of retrieved retain candidates
        """
        retrieved_retains = []
        if facts == []:
            return retrieved_retains

        if len(facts) == 32:
            query = query + "/n/n"

        _, probs, pf_pairs = self.retrieve_facts_list(query, facts)
        index = torch.argmax(probs)
        retains = list(pf_pairs.keys())
        cand = retains[index]

        if cand == "STOP":
            return retrieved_retains

        retrieved_retains = [cand]
        facts.remove(cand)
        return retrieved_retains + self.construct_retrieved_examples(query + cand, facts)

    def __get_embeddings(self, text):
        """
        Helper to get BERT embeddings for a list of strings.
        """
        if isinstance(text, str):
            text = [text]

        device = self.stop_embedding.device
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)

        with torch.no_grad():
            outputs = self.bert(**inputs)

        return outputs.last_hidden_state[:, 0, :]

def sample_retrieved_examples(retriever, query, icl_examples, max_candidates):
    """
    Sample retrieved examples based on current policy
    """
    max_candidates = max_candidates - 1

    if max_candidates == 31:
        query = query + "\n\n"

    _, probs, pairs = retriever.retrieve_facts_list(query, icl_examples)
    m = torch.distributions.Categorical(probs)

    action_idx = m.sample()
    candidates = list(pairs.keys()) # May sample STOP action
    action = candidates[action_idx]
    log_prob = m.log_prob(action_idx).detach()

    if action == "STOP":
        return [("STOP", log_prob)]

    if max_candidates <= 0:
        return [(action, log_prob)]

    query = query + action
    icl_examples.remove(action)

    return [(action, log_prob)] + sample_retrieved_examples(retriever, query, icl_examples, max_candidates)

