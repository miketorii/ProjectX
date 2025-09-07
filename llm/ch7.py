import json
import os
import urllib
import urllib.request

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
    
    
    
