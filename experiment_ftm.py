import json
import torch

from transformers import AutoTokenizer
from easyeditor import BaseEditor, FTHyperParams

HYPERPARAMS = "./hparams/FT/gpt2-xl"
COUNTERFACT_PATH = "./data/counterfact.json"
RESULTS_PATH = "./results/ftm_counterfact.json"
N_EXAMPLES = 30

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_counterfact(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
    
def prepare_counterfact_example(example):
    rewrite = example["requested_rewrite"]

    return {
        "case_id": example["case_id"],
        "subject": rewrite["subject"],
        "prompt": rewrite["prompt"].format(rewrite["subject"]),
        "ground_truth": rewrite["target_true"]["str"],
        "target_new": rewrite["target_new"]["str"],
        "paraphrase_prompts": example["paraphrase_prompts"],
        "neighborhood_prompts": example["neighborhood_prompts"],
        "attribute_prompts": example["attribute_prompts"],
        "generation_prompts": example["generation_prompts"],
    }

def generate_answer(model, tokenizer, prompt, device):
    inputs = tokenizer(prompt, return_tensors="pt")

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    return tokenizer.decode(
        output[0],
        skip_special_tokens=True
    )


def evaluate_prompts(model, tokenizer, prompts, device):
    results = []

    for prompt in prompts:
        answer = generate_answer(
            model,
            tokenizer,
            prompt,
            device
        )

        results.append({
            "prompt": prompt,
            "answer": answer
        })

    return results

def restore_weights(model, weights_copy):
    with torch.no_grad():
        model_parameters = dict(model.named_parameters())

        for name, original_weight in weights_copy.items():
            parameter = model_parameters[name]
            parameter.copy_(original_weight.to(parameter.device))

def check_weights_restored(model, weights_copy):
    model_parameters = dict(model.named_parameters())

    all_restored = True

    for name, original_weight in weights_copy.items():
        current_weight = model_parameters[name]

        same = torch.equal(
            current_weight.cpu(),
            original_weight.cpu()
        )

        max_difference = (
            current_weight.cpu() - original_weight.cpu()
        ).abs().max().item()

        print(
            f"{name}: "
            f"equal={same}, "
            f"max_difference={max_difference}"
        )

        if not same:
            all_restored = False

    return all_restored

def make_result(example,metrics,answer_before,answer_after,paraphrases_before,paraphrases_after,neighborhood_before,neighborhood_after,restored,restore_max_difference):
    metric = metrics[0]

    return {
        "case_id": example["case_id"],
        "subject": example["subject"],
        "prompt": example["prompt"],
        "ground_truth": example["ground_truth"],
        "target_new": example["target_new"],

        "metrics": {
            "pre_rewrite_acc": float(
                metric["pre"]["rewrite_acc"][0]
            ),
            "post_rewrite_acc": float(
                metric["post"]["rewrite_acc"][0]
            ),
            "pre_rephrase_acc": float(metric["pre"]["rephrase_acc"][0]),
            "post_rephrase_acc": float(metric["post"]["rephrase_acc"][0]),
            "post_neighborhood_acc": float(metric["post"]["locality"]["neighborhood_acc"][0])
        },

        "main_prompt": {
            "before": answer_before,
            "after": answer_after
        },

        "paraphrases": {
            "before": paraphrases_before,
            "after": paraphrases_after
        },

        "neighborhood": {
            "before": neighborhood_before,
            "after": neighborhood_after
        },

        "restoration": {
            "restored": restored,
            "max_difference": restore_max_difference
        }
    }

counterfact = load_counterfact(COUNTERFACT_PATH)
examples = [prepare_counterfact_example(counterfact[i]) for i in range(N_EXAMPLES)]
all_results = []

tokenizer = AutoTokenizer.from_pretrained("gpt2-xl")
hparams = FTHyperParams.from_hparams(HYPERPARAMS)
hparams.device = device
editor = BaseEditor.from_hparams(hparams)

for example in examples:
    print("Case ID:", example["case_id"])

#BEFORE EDIT
    answer_before = generate_answer(
        editor.model,
        tokenizer,
        example["prompt"],
        device
    )
#PARAPHRASES BEFORE
    paraphrase_prompt = [example["paraphrase_prompts"][0]]
    paraphrases_before = evaluate_prompts(
        editor.model,
        tokenizer,
        paraphrase_prompt,
        device
    )
#NEIGHBORHOOD BEFORE
    neighborhood_prompt = [example["neighborhood_prompts"][0]]
    neighborhood_before = evaluate_prompts(
        editor.model,
        tokenizer,
        neighborhood_prompt,
        device
    )
#FT-M EDIT
    locality_inputs = {
    "neighborhood": {
        "prompt": example["neighborhood_prompts"][0],
        "ground_truth": example["ground_truth"]
    }
}
    metrics, edited_model, weights_copy = editor.edit(
        prompts=[example["prompt"]],
        ground_truth=[example["ground_truth"]],
        target_new=[example["target_new"]],
        rephrase_prompts=[example["paraphrase_prompts"][0]],
        locality_inputs=locality_inputs,
        keep_original_weight=True,
        sequential_edit=True
    )


#AFTER EDIT
    answer_after = generate_answer(
        edited_model,
        tokenizer,
        example["prompt"],
        device
    )

#PARAPHRASES AFTER
    paraphrases_after = evaluate_prompts(
        edited_model,
        tokenizer,
        paraphrase_prompt,
        device
    )

#NEIGHBORHOOD AFTER
    neighborhood_after = evaluate_prompts(
        edited_model,
        tokenizer,
        neighborhood_prompt,
        device
    )

#RESTORE
    restore_weights(
        edited_model,
        weights_copy
    )
    restored = check_weights_restored(
        edited_model,
        weights_copy
    )

#AFTER RESTORE
    answer_restored = generate_answer(
        edited_model,
        tokenizer,
        example["prompt"],
        device
    )
    restore_max_difference = 0.0
    model_parameters = dict(edited_model.named_parameters())
    for name, original_weight in weights_copy.items():
        current_weight = model_parameters[name]
        difference = (current_weight.cpu() - original_weight.cpu()).abs().max().item()
        restore_max_difference = max(restore_max_difference,difference)

#Save results
    result = make_result(
        example=example,
        metrics=metrics,
        answer_before=answer_before,
        answer_after=answer_after,
        paraphrases_before=paraphrases_before,
        paraphrases_after=paraphrases_after,
        neighborhood_before=neighborhood_before,
        neighborhood_after=neighborhood_after,
        restored=restored,
        restore_max_difference=restore_max_difference
    )

    all_results.append(result)
with open(RESULTS_PATH,"w",encoding="utf-8") as f:
    json.dump(all_results,f,ensure_ascii=False,indent=4)

print("RESULTS SAVED")
