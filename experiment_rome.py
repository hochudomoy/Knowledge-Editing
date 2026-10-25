import json
import torch
import numpy as np
from collections import defaultdict
from transformers import AutoTokenizer
from easyeditor import BaseEditor, ROMEHyperParams

HYPERPARAMS = "./hparams/ROME/gpt2-xl"
COUNTERFACT_PATH = "./data/counterfact.json"
LAYER = 0
RESULTS_PATH = f"./results/rome_counterfact_{LAYER}.json"
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

def make_result(example,initial_knowledge,metrics,counterfact_metrics,answer_before,answer_after,paraphrases_before,paraphrases_after,neighborhood_before,neighborhood_after,restoration_after,restored,restore_max_difference):
    metric = metrics[0]

    return {
        "case_id": example["case_id"],
        "subject": example["subject"],
        "prompt": example["prompt"],
        "ground_truth": example["ground_truth"],
        "target_new": example["target_new"],
        "pre_edit_old_fact_check":{"old_fact_preferred": initial_knowledge['rewrite'],
                                   "paraphrase_old_fact_preferred_count": initial_knowledge['paraphrase_count'],
                                   "paraphrase_total": initial_knowledge['paraphrase_total'],
                                   "paraphrase_old_fact_preferred_ratio":initial_knowledge['ratio']},

        "metrics": {
             "pre": {"rewrite_acc": float(metric["pre"]["rewrite_acc"][0]),
                     "rephrase_acc": float(metric["pre"]["rephrase_acc"][0]),},
            "post":{"rewrite_acc": float(metric["post"]["rewrite_acc"][0]),
                    "rephrase_acc": float(metric["post"]["rephrase_acc"][0]),
                    "neighborhood_acc": float(metric["post"]["locality"]["neighborhood_acc"][0])},
                    },
        "counterfact_metrics": {
            "pre": {
                    "rewrite_success": counterfact_metrics["pre"]["rewrite_success"],
                    "paraphrase_success": counterfact_metrics["pre"]["paraphrase_success"],
                    "neighborhood_success": counterfact_metrics["pre"]["neighborhood_success"],
                    "rewrite_diff":  counterfact_metrics["pre"]["rewrite_diff"],
                    "paraphrase_diff":  counterfact_metrics["pre"]["paraphrase_diff"],
                    "neighborhood_diff":  counterfact_metrics["pre"]["neighborhood_diff"],
                    },
            "post": {
                    "rewrite_success": counterfact_metrics["post"]["rewrite_success"],
                    "paraphrase_success": counterfact_metrics["post"]["paraphrase_success"],
                    "neighborhood_success": counterfact_metrics["post"]["neighborhood_success"],
                    "rewrite_diff":  counterfact_metrics["post"]["rewrite_diff"],
                    "paraphrase_diff":  counterfact_metrics["post"]["paraphrase_diff"],
                    "neighborhood_diff":  counterfact_metrics["post"]["neighborhood_diff"],
                    }
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
            "answer_restored": restoration_after,
            "restored": restored,
            "max_difference": restore_max_difference
        }
    }

def test_prediction(model, tok, prefixes, target_new, target_true, device):
    prefix_lens = [len(n) for n in tok(prefixes)["input_ids"]]
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    
    prompt_tok = tok([
        f"{prefix} {answer}"
        for prefix in prefixes
        for answer in [target_new, target_true]
    ],
    padding=True,
    return_tensors="pt",
    ).to(device)

    new_ids = tok(f" {target_new}")["input_ids"]
    true_ids = tok(f" {target_true}")["input_ids"]

    with torch.no_grad():
        logits = model(**prompt_tok).logits

    results = np.zeros((logits.size(0),), dtype=np.float32)

    for i in range(logits.size(0)):
        cur_len = len(new_ids) if i % 2 == 0 else len(true_ids)

        for j in range(cur_len):
            cur_tok=(new_ids if i%2 ==0 else true_ids)[j]
            results[i] += -torch.nn.functional.log_softmax(
                logits[i, prefix_lens[i // 2] + j - 1, :], dim=0)[cur_tok].item()

        results[i]/=cur_len

    return [
         {"target_new": results[i].item(),
          "target_true": results[i + 1].item(),}
        for i in range(0, len(results), 2)
    ]

def evaluate_counterfact_group(model,tokenizer,prompts,target_new,target_true,device,group):
    probs = test_prediction(model=model,tok=tokenizer,prefixes=prompts,target_new=target_new,target_true=target_true,device=device)

    if group in ["rewrite", "paraphrase"]:
        success = np.mean([item["target_true"] > item["target_new"] for item in probs])
        diff = np.mean([np.exp(-item["target_new"]) - np.exp(-item["target_true"]) for item in probs])

    elif group == "neighborhood":
        success = np.mean([item["target_true"] < item["target_new"] for item in probs])
        diff = np.mean([np.exp(-item["target_true"]) - np.exp(-item["target_new"])for item in probs])
    return {
        "probs": probs,
        "success": float(success),
        "diff": float(diff),
    }


def evaluate_counterfact_metrics(model,tokenizer,example,device):
    groups = {
        "rewrite": [example["prompt"]],
        "paraphrase": example["paraphrase_prompts"],
        "neighborhood": example["neighborhood_prompts"]
    }

    metrics = {}

    for group, prompts in groups.items():
        result = evaluate_counterfact_group(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            target_new=example["target_new"],
            target_true=example["ground_truth"],
            device=device,
            group=group
        )

        metrics[f"{group}_prompts_probs"] = result["probs"]
        metrics[f"{group}_success"] = result["success"]
        metrics[f"{group}_diff"]=result["diff"]

    return metrics


counterfact = load_counterfact(COUNTERFACT_PATH)
examples = [prepare_counterfact_example(counterfact[i]) for i in range(N_EXAMPLES)]
all_results = []

tokenizer = AutoTokenizer.from_pretrained("gpt2-xl")
hparams = ROMEHyperParams.from_hparams(HYPERPARAMS)
hparams.layers = [LAYER]
hparams.device = device
editor = BaseEditor.from_hparams(hparams)

for example in examples:
    print("Case ID:", example["case_id"])

    print('BEFORE EDIT')
    answer_before = generate_answer(
        editor.model,
        tokenizer,
        example["prompt"],
        device
    )
    counterfact_metrics_pre = evaluate_counterfact_metrics(model=editor.model,tokenizer=tokenizer,example=example,device=device)
    rewrite_probs = counterfact_metrics_pre["rewrite_prompts_probs"]
    paraphrase_probs = counterfact_metrics_pre["paraphrase_prompts_probs"]
    initial_knowledge={'rewrite': rewrite_probs[0]["target_true"] < rewrite_probs[0]["target_new"],
                       'paraphrase_count': sum(item["target_true"] < item["target_new"] for item in paraphrase_probs),
                       'paraphrase_total':len(paraphrase_probs),
                       'ratio':sum(item["target_true"] < item["target_new"] for item in paraphrase_probs)/len(paraphrase_probs),
                       }                 
#PARAPHRASES BEFORE
    paraphrases_before = evaluate_prompts(
        editor.model,
        tokenizer,
        example["paraphrase_prompts"],
        device
    )

#NEIGHBORHOOD BEFORE
    neighborhood_before = evaluate_prompts(
        editor.model,
        tokenizer,
        example["neighborhood_prompts"],
        device
    )
#ROME EDIT
    locality_inputs = {
    "neighborhood": {
        "prompt": example["neighborhood_prompts"][0],
        "ground_truth": example["ground_truth"]
    }
}
    metrics, edited_model, weights_copy = editor.edit(
        prompts=[example["prompt"]],
        subject=[example["subject"]],
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
    counterfact_metrics_post = evaluate_counterfact_metrics(model=edited_model,tokenizer=tokenizer,example=example,device=device)
    counterfact_metrics = {"pre": counterfact_metrics_pre,
                           "post": counterfact_metrics_post,}
#PARAPHRASES AFTER
    paraphrases_after = evaluate_prompts(
        edited_model,
        tokenizer,
        example["paraphrase_prompts"],
        device
    )

#NEIGHBORHOOD AFTER
    neighborhood_after = evaluate_prompts(
        edited_model,
        tokenizer,
        example["neighborhood_prompts"],
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
    answer_matches_before = answer_restored == answer_before
    print(answer_matches_before)
    restore_max_difference = 0.0
    model_parameters = dict(edited_model.named_parameters())
    for name, original_weight in weights_copy.items():
        current_weight = model_parameters[name]
        difference = (current_weight.cpu() - original_weight.cpu()).abs().max().item()
        restore_max_difference = max(restore_max_difference,difference)

#Save results
    result = make_result(
        example=example,
        initial_knowledge=initial_knowledge,
        metrics=metrics,
        counterfact_metrics=counterfact_metrics,
        answer_before=answer_before,
        answer_after=answer_after,
        paraphrases_before=paraphrases_before,
        paraphrases_after=paraphrases_after,
        neighborhood_before=neighborhood_before,
        neighborhood_after=neighborhood_after,
        restoration_after=answer_restored,
        restored=restored,
        restore_max_difference=restore_max_difference
    )

    all_results.append(result)

knowledge_groups = defaultdict(list)
for result in all_results:
    check = result["pre_edit_old_fact_check"]
    main_old_fact = check["old_fact_preferred"]
    paraphrase_ratio = check["paraphrase_old_fact_preferred_ratio"]
    ground_truth = result["ground_truth"].lower()
    generation_main_old = (ground_truth in result["main_prompt"]["before"].lower())
    generation_paraphrase_count = sum(ground_truth in item["answer"].lower()for item in result["paraphrases"]["before"])
    generation_paraphrase_ratio = (generation_paraphrase_count / len(result["paraphrases"]["before"]))


    if (generation_main_old and generation_paraphrase_ratio >= 0.5) and (main_old_fact and paraphrase_ratio >= 0.5):
        category = "old_fact_reproduced"
    elif not (main_old_fact and paraphrase_ratio >= 0.5) and not (generation_main_old and generation_paraphrase_ratio >= 0.5):
        category = "old_fact_not_reproduced"
    else:
        category = "inconsistent"

    result["initial_knowledge_category"] = category
    knowledge_groups[category].append(result)

counterfact_metric_summary = {}
for stage in ["pre", "post"]:
    counterfact_metric_summary[stage] = {}
    for metric_name in ["rewrite_success","paraphrase_success","neighborhood_success"]:
        values = [result["counterfact_metrics"][stage][metric_name]for result in all_results]
        counterfact_metric_summary[stage][metric_name] = float(np.mean(values))

knowledge_group_summary = {}
for category, results in knowledge_groups.items():
    knowledge_group_summary[category]={}
    for stage in ("pre", "post"):
        knowledge_group_summary[category][stage]={}
        for metric in ("rewrite_success","paraphrase_success","neighborhood_success"):
            values = [result["counterfact_metrics"][stage][metric] for result in results]
            knowledge_group_summary[category][stage][metric] = (float(np.mean(values)) if values else None)

diff_group_summary = {}
for category, results in knowledge_groups.items():
    diff_group_summary[category] = {}
    for stage in ("pre", "post"):
        diff_group_summary[category][stage] = {}
        for metric in ("rewrite_diff","paraphrase_diff","neighborhood_diff"):
            values = [result["counterfact_metrics"][stage][metric] for result in results]
            diff_group_summary[category][stage][metric] = (float(np.mean(values)) if values else None)
with open(RESULTS_PATH,"w",encoding="utf-8") as f:
    json.dump( {"results": all_results,"counterfact_metric_summary": counterfact_metric_summary, "knowledge_group_summary":knowledge_group_summary,"diff_group_summary": diff_group_summary},f,ensure_ascii=False,indent=4, )

print("RESULTS SAVED")
