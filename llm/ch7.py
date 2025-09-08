import json
import os
import urllib
import urllib.request

import torch
from torch.utils.data import Dataset

import tiktoken

#######################################
#
#
def download_and_load_file(file_path, url):
    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:            
            text_data = response.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data

#######################################
#
#
def format_input(entry):
    instruction_text = (
        f"Below is an instruction"
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input: \n{entry['input']}" if entry['input'] else ""
    
    return instruction_text + input_text

#######################################
#
#
class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data

        self.encoded_texts = []
        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response: \n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(
                tokenizer.encode(full_text)
            )

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)
    
#######################################
#
#
if __name__ == "__main__":

    #################################################3
    ##
    file_path = "instruction-data.json"
    url = (
        "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
        "/main/ch07/01_main-chapter-code/instruction-data.json"
    )
    data = download_and_load_file(file_path, url)
    print("Number of entries:", len(data))

    print("Example: ", data[50])
    model_input = format_input(data[50])
    desired_response = f"\n\n### Response: \n{data[50]['output']}"
    print(model_input + desired_response)
    
    train_portion = int(len(data) * 0.85)
    test_portion = int(len(data) * 0.1)
    val_portion = len(data) - train_portion - test_portion

    train_data = data[:train_portion]
    test_data = data[train_portion:train_portion+test_portion]
    val_data = data[train_portion+test_portion:]

    print("Training set length:", len(train_data))
    print("Test set length:", len(test_data))
    print("Validation set length:", len(val_data))    
    
    #################################################3
    ##
    tokenizer = tiktoken.get_encoding("gpt2")
    print(tokenizer.encode( "<|endoftext|>", allowed_special={"<|endoftext|>"} ))
    
