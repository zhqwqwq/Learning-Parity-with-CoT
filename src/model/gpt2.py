from transformers import AutoModelForCausalLM, AutoConfig, LlamaForCausalLM, AutoTokenizer
import torch

# PATH = 'gpt2'
PATH = 'huggingface_transformer_model/gpt2'

def get_gpt_transformer_config(
        vocab_size = 50264,
        hidden_size=768,
        intermediate_size=1024,
        num_hidden_layers=12,
        num_attention_heads=12,
        max_position_embeddings=4096,
):
    config = AutoConfig.from_pretrained(PATH)
    config.vocab_size = vocab_size 
    config.hidden_size = hidden_size
    config.intermediate_size = intermediate_size
    config.num_hidden_layers = num_hidden_layers
    config.num_attention_heads = num_attention_heads
    config.num_key_value_heads = num_attention_heads
    config.max_position_embeddings = max_position_embeddings
    return config


def get_gpt_transformer_model_from_config(
        onehot_embed = False,
        wpe_train = False,
        vocab_size = 50264,
        hidden_size=32,
        intermediate_size=128,
        num_hidden_layers=16,
        num_attention_heads=4,
        max_position_embeddings=4096
    ):
    config = get_gpt_transformer_config(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        max_position_embeddings=max_position_embeddings
    )
    model = AutoModelForCausalLM.from_config(config) 
    if onehot_embed:
        model.transformer.wte = torch.nn.Embedding(vocab_size, hidden_size, dtype = torch.bfloat16)
        model.transformer.wpe.weight.data.requires_grad = True
        manual_embed = torch.nn.init.eye_(torch.empty(vocab_size, hidden_size)).to(dtype = torch.bfloat16)
        manual_embed.requires_grad = False
        model.transformer.wte.weight.data = manual_embed
        model.lm_head.weight.data = manual_embed.clone()
    else:
        model.lm_head.weight.data = model.lm_head.weight.data.clone()
    if wpe_train:
        model.transformer.wpe.weight.data.requires_grad = True
    return model

def get_gpt_transformer_model(
    from_config,
    onehot_embed = False,
    wpe_train = False, 
    vocab_size = 50264,
    hidden_size=32,
    intermediate_size=128,
    num_hidden_layers=16,
    num_attention_heads=4,
    max_position_embeddings=4096
):
    if from_config:
        model = get_gpt_transformer_model_from_config(
            onehot_embed = onehot_embed,
            wpe_train = wpe_train,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            max_position_embeddings=max_position_embeddings
        ).to(dtype = torch.bfloat16)
    else:
        model = LlamaForCausalLM.from_pretrained(PATH, torch_dtype = torch.bfloat16, resume_download = True)
    print(f'Transformer Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e9:.2f}B')
    return model

    
if __name__ == '__main__':
    # pretrained
    # model = get_transformer_model(from_config = False)
    # tokenizer = AutoTokenizer.from_pretrained(PATH)
    # tokenized_input = tokenizer("Hello, who are you?", return_tensors = 'pt')
    # input_ids = tokenized_input['input_ids']
    # attention_mask = tokenized_input['attention_mask']
    # output = model(input_ids, attention_mask = attention_mask)
    # scratch
    model = get_gpt_transformer_model(from_config = True, onehot_embed = True, vocab_size = 2)  
    from IPython import embed; embed()
    