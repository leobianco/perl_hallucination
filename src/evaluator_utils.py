import numpy as np
import torch
from datasets import concatenate_datasets
from tqdm import tqdm


def get_fewshot_examples(data, n_yes: int, n_no: int, seed: int):
    if n_yes == 0 and n_no == 0:
        return None
    positive_examples = (
        data.filter(lambda entry: entry["class_hall"] == "Yes")
        .shuffle(seed=seed)
        .select(range(n_yes))
    )
    negative_examples = (
        data.filter(lambda entry: entry["class_hall"] == "No")
        .shuffle(seed=seed)
        .select(range(n_no))
    )
    fewshot_examples = (
        concatenate_datasets([positive_examples, negative_examples]).shuffle(
            seed=seed
        )
        if positive_examples is not None
        else None
    )
    return fewshot_examples


def evaluator_score_batch(
    evaluator, tokenized_prompts, yes_token_id, no_token_id
):
    with torch.no_grad():
        outputs = evaluator(**tokenized_prompts, use_cache=False)
        score_yes = torch.exp(outputs.logits[:, -1, yes_token_id])
        score_no = torch.exp(outputs.logits[:, -1, no_token_id])
        score_batch = score_no / (score_yes + score_no)
    return score_batch


def evaluator_score(
    data, script_args, tokenizer, evaluator, yes_token_id, no_token_id
):
    iterator = data.iter(batch_size=script_args.eval_batch_size)
    num_batches = int(data.num_rows / script_args.eval_batch_size)
    scores = torch.tensor([])
    for batch in tqdm(iterator, desc="Evaluator scoring", total=num_batches):
        tokenized_prompts = tokenizer(
            batch["evaluator_prompt"], return_tensors="pt", padding="longest"
        )
        tokenized_prompts = {
            k: v.to(evaluator.device) for k, v in tokenized_prompts.items()
        }
        score_batch = evaluator_score_batch(
            evaluator, tokenized_prompts, yes_token_id, no_token_id
        )
        score_batch = score_batch.cpu()
        scores = torch.cat((scores, score_batch))
        del tokenized_prompts
    return scores


def gemini_score_response(response):
    if response.candidates[0].avg_logprobs is None:
        return 0
    if response.text == "No":
        return np.exp(response.candidates[0].avg_logprobs)
    elif response.text == "Yes":
        return 1 - np.exp(response.candidates[0].avg_logprobs)
    else:
        raise Exception("Invalid response")


def gemini_score_dataset(client, dataset, script_args):
    model = "gemini-2.0-flash-001"
    scores = []
    for entry in dataset:
        resp = client.models.generate_content(
            model=model, contents=entry["evaluator_prompt"]
        )
        scores.append(gemini_score_response(resp))
    return torch.tensor(scores)
