import torch
from transformers import AutoTokenizer, pipeline

def load_model(model_id, device=None):
    """
    Loads frozen LLM
    """
    tokenizer1 = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer1.pad_token = tokenizer1.eos_token

    pipeline = pipeline(
        "text-generation",
        model=model_id,
        tokenizer=tokenizer1,
        model_kwargs={"dtype": torch.bfloat16},
        device_map="auto",
        trust_remote_code=True
    )

    frozen_model = pipeline.model
    frozen_model.eval()
    for param in frozen_model.parameters():
        param.requires_grad = False

    return pipeline

def generate_response(prompt, temperature=0.0, top_p=0.9, top_k=50):
    """
    Calls the LLM and returns the last word of the response.
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant who will complete the last sentence with a single word."},
        {"role": "user", "content": "Using the following information complete the last sentence\n" + prompt}
    ]

    outputs = pipeline(
        messages,
        max_new_tokens=32,
        batch_size=1,

        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
        pad_token_id=pipeline.tokenizer.eos_token_id,

        return_full_text=False
    )

    try:
        generated_text = outputs[0]['generated_text']

        if isinstance(generated_text, list):
             generated_text = generated_text[-1]['content']
        elif isinstance(generated_text, dict):
             generated_text = generated_text['content']

        llm_answer = generated_text.strip().split()[-1].strip(".").strip()
        return llm_answer

    except Exception as e:
        print(f"Error parsing output: {e}")
        print(f"Raw output was: {outputs}")
        return None
    
def compute_reward(llm_answer, ground_truth):
    """
    Compute binary reward based on the LLM response and the expected true answer
    """
    return 1 if llm_answer.lower().strip() == ground_truth.strip().lower() else 0
