from .gpt2 import get_gpt_transformer_model

def get_model(
    model_type,
    **kwargs
):
    assert model_type == 'gpt2'
    model = get_gpt_transformer_model(
        **kwargs
    )
    print(f'Model Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e9:.2f}B')
    return model

