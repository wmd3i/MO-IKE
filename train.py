"""
MO-IKE trianing for retriever
"""
import torch
from copy import deepcopy
import torch.optim as optim
from utils.llm_utils import load_model

from copy import deepcopy
import torch
import torch.optim as optim
from tqdm import tqdm


# -----------------------------
# Training Setup
# -----------------------------

# Define hyperparameters
num_epoches = 1
num_inner_epoch = 1
training_size = 300
group_size = 8
epsilon = 0.2
beta = 0.05

optimizer = optim.Adam(
    list(retriever.query_actor.parameters()) +
    list(retriever.cand_actor.parameters()) +
    [retriever.stop_embedding, retriever.stop_bias],
    lr=1e-5
)
retriever.train()

# Define ref_retriever
ref_retriever = deepcopy(retriever)
for param in ref_retriever.parameters():
    param.requires_grad = False
ref_retriever.eval()

K = 16

# -----------------------------
# Training Loop with MO-IKE
# -----------------------------
for epoch in range(num_epoches):
    epoch_actor_loss = 0.0
    correct_edits = 0 # Count correct edits
    total_predictions = 0 # Total number of LLM calls used for reward

    print(f"Epoch {epoch + 1}")
    for example_idx in tqdm(range(training_size)):
        i = training_data[example_idx]
        line=lines[i]

        relation = line['requested_rewrite']['relation_id']
        prompt = line['requested_rewrite']['prompt']
        subject = line['requested_rewrite']['subject']
        prompt_calibrate = prompt.format('SUBJECT')
        prompt = prompt.format(subject)
        paraphrase = line['paraphrase_prompts'][0]
        neighbor= line['neighborhood_prompts'][0]
        PROMPTS = [prompt, prompt_calibrate]

        target_true = line['requested_rewrite']['target_true']['str']
        target_new = line['requested_rewrite']['target_new']['str']

        targets = [target_new, target_true]
        ground_truth = targets[0]
        query = line['requested_rewrite']['prompt'].format(line['requested_rewrite']['subject'])
        icl_copy, icl_update, icl_retain = construct_icl_examples(idx=i, demos=demos, order=order)
        icl_examples = icl_copy + icl_update + icl_retain

        # Construct group rollouts and rewards
        group = []
        group_rewards = []
        for _ in range(group_size):
            sample_examples = icl_examples.copy()
            eps = sample_retrieved_retains(query, sample_examples, 32)
            group.append(eps)
            retrieved_examples = [cand[0] for cand in eps]
            # Truncate STOP signal
            if retrieved_examples[-1] == 'STOP':
                retrieved_examples = retrieved_examples[:-1]

            # Compute reward based on each metrics
            icl_edit, icl_para, icl_retain = [deepcopy(retrieved_examples) for i in range(3)]
            icl_edit.append(f'New Fact: {prompt} {target_new}\nPrompt: {prompt} {target_new}\n\n')
            icl_edit.append(f'Prompt: {prompt}\n')
            prompts_edit = "".join(icl_edit)
            edit_answer = generate_response(prompts_edit)
            edit_reward = compute_reward(edit_answer, ground_truth)

            icl_para.append(f'New Fact: {prompt} {target_new}\nPrompt: {prompt} {target_new}\n\n')
            icl_para.append(f'Prompt: {paraphrase}')
            prompts_para = "".join(icl_para)
            para_answer = generate_response(prompts_para)
            para_reward = compute_reward(para_answer, ground_truth)

            icl_retain.append(f'New Fact: {prompt} {target_new}\nPrompt: {prompt} {target_new}\n\n')
            icl_retain.append(f'Prompt: {neighbor}')
            prompts_retain = "".join(icl_retain)
            retain_answer = generate_response(prompts_retain)
            retain_reward = compute_reward(retain_answer, target_true)

            final_reward = edit_reward + para_reward + retain_reward

            group_rewards.append(final_reward)

        rewards_tensor = torch.tensor(group_rewards, dtype=torch.float32).to(device)
        mean = rewards_tensor.mean(dim=-1, keepdim=True)
        variance = ((rewards_tensor - mean) ** 2).mean(dim=-1, keepdim=True)
        std = (variance + 1e-8).sqrt()
        advantages = (rewards_tensor - mean) / std
        print(group_rewards)
        print(mean, variance, advantages)

        for inner_epoch in range(num_inner_epoch):
            total_loss = 0.0
            optimizer.zero_grad()

            for g_idx in range(group_size):
                trajectory = group[g_idx]
                advantage = advantages[g_idx]
                curr_icl_examples = icl_examples.copy()
                traj_loss = 0.0
                curr_query = line['requested_rewrite']['prompt'].format(line['requested_rewrite']['subject'])

                for t, (action, old_log_prob) in enumerate(trajectory):
                    _, probs, pairs = retriever.retrieve_facts_list(curr_query, curr_icl_examples)
                    
                    # --- NEW: Fetch reference probabilities ---
                    with torch.no_grad():
                        _, ref_probs, _ = ref_retriever.retrieve_facts_list(curr_query, curr_icl_examples)
                    # ------------------------------------------

                    candidates = list(pairs.keys())
                    if action not in candidates:
                        break
                    action_idx = candidates.index(action)
                    
                    new_d = torch.distributions.Categorical(probs)
                    ref_d = torch.distributions.Categorical(ref_probs)
                    
                    new_log_prob = new_d.log_prob(torch.tensor(action_idx).to(device))

                    ratio = torch.exp(new_log_prob - old_log_prob)
                    surr1 = ratio * advantage
                    surr2 = torch.clamp(ratio, 1-epsilon, 1+epsilon) * advantage
                    
                    loss = -torch.min(surr1, surr2)
                    kl_div = torch.distributions.kl.kl_divergence(new_d, ref_d)
                    step_loss = loss + (beta * kl_div)
                    
                    traj_loss += step_loss

                    if action == 'STOP':
                        break

                    if action in curr_icl_examples:
                        curr_icl_examples.remove(action)

                    if t == 0:
                        curr_query = curr_query + '\n\n'
                    curr_query = curr_query + action

                total_loss += traj_loss

            final_loss = total_loss / group_size
            if isinstance(final_loss, torch.Tensor) and final_loss.requires_grad:
                final_loss.backward()
                torch.nn.utils.clip_grad_norm_(retriever.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_actor_loss += final_loss.item()
            print(final_loss)

print(f"Avg Loss: {epoch_actor_loss / 300}")

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    id = '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'
    load_model(id, model_id='meta-llama/Llama-3.2-3B-Instruct', device=device)