def construct_icl_examples(idx, demos, order, corpus_idx):
    """
    Initial retrieval based on IKE method mainly
    """
    icl_copy = []
    icl_update = []
    icl_retain = []

    demo_ids = corpus_idx[idx]
    demo_ids = demo_ids[:len(order)]

    for demo_id, o in zip(demo_ids, order):
        line = demos[demo_id - 2000]  # Adjusting index to match demos
        new_fact = line['requested_rewrite']['prompt'].format(line['requested_rewrite']['subject'])
        target_new = line['requested_rewrite']['target_new']['str']
        target_true = line['requested_rewrite']['target_true']['str']

        if o == 0:  # Copy
            icl_copy.append(
                f"New Fact: {new_fact} {target_new}\nPrompt: {new_fact} {target_new}\n\n"
            )
        elif o == 1:  # Update
            prompt = line['paraphrase_prompts'][0]
            icl_update.append(
                f"New Fact: {new_fact} {target_new}\nPrompt: {prompt} {target_new}\n\n"
            )
        elif o == 2:  # Retain
            prompt = line['neighborhood_prompts'][0]
            icl_retain.append(
                f"New Fact: {new_fact} {target_new}\nPrompt: {prompt} {target_true}\n\n"
            )
            
            
    return icl_copy, icl_update, icl_retain