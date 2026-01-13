import hashlib

from datasets import load_dataset, DatasetDict

dataset = load_dataset("hkust-nlp/deita-10k-v0", split="train")


def format_prompt(example):
    convs = example["conversations"]
    if convs[0]["from"] == "gpt":
        convs = convs[1:]

    prompt = convs[0]["value"]

    messages = []
    for i, message in enumerate(convs):
        if message["from"] == "human" and i % 2 == 0:
            messages.append({"role": "user", "content": message["value"]})
        elif message["from"] == "gpt" and i % 2 == 1:
            messages.append({"role": "assistant", "content": message["value"]})
        else:
            raise ValueError("Unknown message type: {} or index mismatch: {}".format(message["from"], i))

    return {
        "prompt": prompt,
        "prompt_id": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "messages": messages,
    }


dataset = dataset.map(format_prompt, remove_columns=dataset.column_names)
dataset = dataset.train_test_split(test_size=0.05, seed=42)  # only 5% examples for test set as small dataset


# remove final assistant turn from rl_ds
def remove_last_turn(x):
    x["messages"] = x["messages"][:-1]
    return x


final_ds = DatasetDict()
final_ds["train_sft"] = dataset["train"]
final_ds["test_sft"] = dataset["test"]
final_ds["train_gen"] = dataset["train"].map(remove_last_turn)
final_ds["test_gen"] = dataset["test"].map(remove_last_turn)


final_ds.push_to_hub("HuggingFaceH4/deita-10k-v0-sft", private=True)
