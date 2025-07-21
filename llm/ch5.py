import torch
import torch.nn as nn
import tiktoken

from previous_chapters import GPTModel, generate_text_simple

############################################
#
#
GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Vocabulary size
#    "context_length": 1024, # Context length
    "context_length": 256, # Context length    
    "emb_dim": 768,         # Embedding dimension
    "n_heads": 12,          # Number of attention heads
    "n_layers": 12,         # Number of layers
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": False       # Query-Key-Value bias
}

############################################
#
#
def calc_text_generation_loss(model, tokenizer):
#    inputs = torch.tensor([
#        [16833, 3626, 6100],
#        [40, 1107, 588]
#    ])
#    targets = torch.tensor([
#        [3626, 6100, 345],
#        [1107, 588, 11311]
#    ])
#
#    with torch.no_grad():
#        logits = model(inputs)
#        probas = torch.softmax(logits, dim=-1)
#        print(probas.shape)

    inputs = torch.tensor([[16833, 3626, 6100],   # ["every effort moves",
                           [40,    1107, 588]])   #  "I really like"]

    targets = torch.tensor([[3626, 6100, 345  ],  # [" effort moves you",
                            [1107,  588, 11311]]) #  " really like chocolate"]
        
    with torch.no_grad():
        logits = model(inputs)
        print(logits)
        
    probas = torch.softmax(logits, dim=-1) # Probability of each token in vocabulary
    print(probas.shape) # Shape: (batch_size, num_tokens, vocab_size)
    print(probas)
        
    token_ids = torch.argmax(probas, dim=-1, keepdim=True)
    print("Token IDs:\n", token_ids)

    print("Target batch 1:\n", token_ids_to_text(targets[0], tokenizer))
    print("Output batch 1:\n", token_ids_to_text(token_ids[0].flatten(), tokenizer))

    text_idx = 0
    target_probas_1 = probas[text_idx, [0,1,2], targets[text_idx]]
    print("Text 1:",target_probas_1)

    text_idx = 1
    target_probas_2 = probas[text_idx, [0,1,2], targets[text_idx]]
    print("Text 2:",target_probas_2)

    log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
    print(log_probas)

    avg_log_probas = torch.mean(log_probas)
    print(avg_log_probas)

    neg_avg_log_probas = avg_log_probas * -1
    print(neg_avg_log_probas)

    
        
############################################
#
#
def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())
    
############################################
#
#
if __name__ == "__main__":
    torch.manual_seed(123)

    model = GPTModel(GPT_CONFIG_124M)
    model.eval()

    start_context = "Every effort moves you"
    tokenizer = tiktoken.get_encoding("gpt2")
    
    token_ids = generate_text_simple(
        model=model,
        idx=text_to_token_ids(start_context, tokenizer),
        max_new_tokens=10,
        context_size=GPT_CONFIG_124M["context_length"]
    )

    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

    print("---------------------------")
    calc_text_generation_loss(model, tokenizer)
       


