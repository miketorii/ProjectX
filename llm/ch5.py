import torch
import torch.nn as nn
import tiktoken

from previous_chapters import GPTModel, generate_text_simple, create_dataloader_v1

import os
import urllib.request

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

def calc_cross_entropy(model, tokenizer):
    print("------------cross entropy------------")
    
    inputs = torch.tensor([[16833, 3626, 6100],   # ["every effort moves",
                           [40,    1107, 588]])   #  "I really like"]

    targets = torch.tensor([[3626, 6100, 345  ],  # [" effort moves you",
                            [1107,  588, 11311]]) #  " really like chocolate"]
        
    with torch.no_grad():
        logits = model(inputs)
        print(logits)

    logits_flat = logits.flatten(0,1)
    targets_flat = targets.flatten()

    print("logits flat:", logits_flat.shape)
    print("targets flat:", targets_flat.shape)

    loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
    print(loss)

    perplexity = torch.exp(loss)
    print(perplexity)

############################################
#
#
def calc_train(model, tokenizer):
    file_path = "the-verdict.txt"

    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()

    print(text_data[:99])
    print(text_data[-99:])    

    total_characters = len(text_data)
    total_tokens = len(tokenizer.encode(text_data))

    print("Characters", total_characters)
    print("Tokens", total_tokens)
                 
    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]

    train_loader = create_dataloader_v1(
        train_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0        
    )

    val_loader = create_dataloader_v1(
        val_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0        
    )    

    if total_tokens * (train_ratio) < GPT_CONFIG_124M["context_length"]:
        print("Not enough token for the training loader")

    if total_tokens * (1-train_ratio) < GPT_CONFIG_124M["context_length"]:
        print("Not enough tokens for the validation loader")

    print("Train loader:")
    for x, y in train_loader:
        print(x.shape, y.shape)

    print("Validation loader:")
    for x, y in val_loader:
        print(x.shape, y.shape)        

    train_tokens = 0
    for input_batch, target_batch in train_loader:
        train_tokens += input_batch.numel()

    val_tokens = 0
    for input_batch, target_batch in val_loader:
        val_tokens += input_batch.numel()

    print("Training tokens;", train_tokens)
    print("Validatiaon tokens:", val_tokens)
    print("All tokens", train_tokens+val_tokens)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    torch.manual_seed(123)

    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, input_batch, target_batch)
        val_loss = calc_loss_loader(val_loader, model, device, input_batch, target_batch)

    print("Training loss:", train_loss)
    print("Validation loss:", val_loss)

############################################
#
#
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy( logits.flatten(0,1), target_batch.flatten() )    
    return loss

############################################
#
#    
#def calc_loss_loader(data_loader, model, device, input_batch, target_batch, num_batches=None):
def calc_loss_loader(data_loader, model, device, num_batches=None):    
    total_loss = 0

    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break

    return total_loss / num_batches
        
############################################
#
#
#if __name__ == "__main__":
def funcmain1():
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
       
    calc_cross_entropy(model, tokenizer)

    calc_train(model, tokenizer)

############################################
#
#
def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer ):

    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)

                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")


        geneate_and_print_sample( model, tokenizer, device, start_context)
    
    return train_losses, val_losses, tokens_seen

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    print("--evaluate model--")
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
        
    model.train()
    return train_loss, val_loss
                
def geneate_and_print_sample( model, tokenizer, device, start_context):
    print("--generate_and_print_sample--")
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size)
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n"," "))
    model.train()
    
    
        
############################################
#
#    
if __name__ == "__main__":
    torch.manual_seed(123)

    model = GPTModel(GPT_CONFIG_124M)
    model.to("cpu")

    tokenizer = tiktoken.get_encoding("gpt2")
    
#######
    file_path = "the-verdict.txt"

    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()

    total_characters = len(text_data)
    total_tokens = len(tokenizer.encode(text_data))

    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]

    train_loader = create_dataloader_v1(
        train_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0        
    )

    val_loader = create_dataloader_v1(
        val_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0        
    )    

######    
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

    device = "cpu"
    num_epochs = 10


    
    print("epoch:", num_epochs)

    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=5, eval_iter=5,
        start_context="Every effort moves you", tokenizer=tokenizer
    )

    print("----------------End--------------")
