from transformers import AutoModelForCausalLM, AutoModel
from transformers import Trainer, EvalPrediction
import transformers
import numpy as np
# import wandb
import sys
import os
import json
import torch
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from utils import set_seed, parse_args
from data import load_dataset
from model import get_model

def compute_entropy(attention_vector):
    entropy = -torch.sum(attention_vector * torch.log(attention_vector + 1e-9), dim=-1)
    return entropy

log_tensor = None
def compute_mean_min_attention_entropy_over_token(attentions):
    layer_head_entropies = []
    global log_tensor
    for layer_attention in attentions:  # (batch_size, num_heads, sequence_length, sequence_length)
        head_entropies = []
        for head_index in range(layer_attention.shape[1]):  # num_heads
            head_attention = layer_attention[:, head_index, :, :]  # (batch_size, sequence_length, sequence_length)
            if log_tensor == None:
                log_tensor = torch.log(torch.arange(2, head_attention.shape[1] + 1, dtype=torch.float)).unsqueeze(0).to(head_attention.device)
            entropy = compute_entropy(head_attention)  # (batch_size,sequence_length)
            normalized_entropy = entropy[:, 1:] = entropy[:, 1:] / log_tensor
            min_entropy_over_token = normalized_entropy.min(dim=1).values
            mean_entropy = min_entropy_over_token.mean().item()
            head_entropies.append(mean_entropy)
        layer_head_entropies.append(head_entropies)
    return layer_head_entropies

class CustomTrainer(Trainer):
    pass
    def evaluation_loop(self, dataloader, description, prediction_loss_only=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        output = super().evaluation_loop(dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix)
        
        all_entropies = []
        for step, inputs in enumerate(dataloader):
            inputs = self._prepare_inputs(inputs)
            with torch.no_grad():
                outputs = self.model(**inputs,output_attentions=True)
                if 'attentions' in outputs:
                    entropies = compute_mean_min_attention_entropy_over_token(outputs['attentions'])
                    all_entropies.append(entropies)

        mean_entropy = torch.tensor(all_entropies).float().mean(dim=0).tolist()
        output.metrics['mean_min_attention_entropy_over_token'] = mean_entropy

        return output

class EarlyStoppingCallback(transformers.TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        eval_accuracy = kwargs.get("metrics")["eval_exact_match"]
        print(eval_accuracy)
        if eval_accuracy and eval_accuracy >= 1.0:
            print(f"Stopping training early. Evaluation accuracy has reached {eval_accuracy}.")
            control.should_training_stop = True

def main():
    args = parse_args()
    set_seed(args.seed)
    train_dataset = load_dataset(args.dataset_dir, args.dataset_type)
    val_dataset = load_dataset(os.path.join(args.dataset_dir, 'val'), args.dataset_type)
    model_args = eval(open(args.model_config_path).read())
    model_args["num_hidden_layers"] = args.num_hidden_layers
    model_args["num_attention_heads"] = args.num_attention_heads
    print(model_args)
    model = get_model(
        **model_args
    )
    if(args.model_dir):
        import safetensors
        safetensors.torch.load_model(model, os.path.join(args.model_dir, 'model.safetensors'))
    output_dir = f"{args.output_dir}{args.dataset_dir.split('/')[-1]}_{args.total_training_samples}_LR={args.lr}_WD={args.weight_decay}_{args.world_size}GPU*{args.batch_size}Batch_{args.model_config_path.split('/')[-1]}_#layer={args.num_hidden_layers}_#head={args.num_attention_heads}"
    training_args = transformers.TrainingArguments(
        output_dir=output_dir,          
        num_train_epochs=args.total_training_samples / len(train_dataset),              
        per_device_train_batch_size=args.batch_size,  
        per_device_eval_batch_size=args.batch_size,   
        warmup_steps=0,                
        weight_decay=args.weight_decay,               
        logging_dir='./logs',            
        logging_steps= args.log_interval // (args.batch_size * args.world_size),
        save_steps = args.save_interval // (args.batch_size *args.world_size),
        save_total_limit = 1,
        evaluation_strategy="steps",     
        eval_steps= args.eval_interval // (args.batch_size * args.world_size), 
        learning_rate = args.lr,
        label_names = ['labels'],
        save_safetensors = False
    )
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        if model_args["model_type"] == 'gpt2_custom_simpler':
            predictions = np.squeeze((predictions >= 0.5).astype(int))
            print(predictions)
        else:
            predictions = np.argmax(predictions, axis=-1)
        predictions = predictions[:, :-1]
        labels = labels[:, 1:]
        exact_match_cnt = 0
        cnt = 0
        for prediction, label in zip(predictions, labels):
            correct = (prediction == label) + (label == -100)
            cnt += 1
            exact_match_cnt += correct.all()
        return {"exact_match": exact_match_cnt / cnt}
    trainer = CustomTrainer(
        model=model, args=training_args, train_dataset=train_dataset,
        compute_metrics=compute_metrics,
        eval_dataset = val_dataset,
    )
    trainer.train(ignore_keys_for_eval = ['past_key_values', 'dreamer_loss_1', 'dreamer_loss_0'])
    trainer.save_model(output_dir=args.output_dir)

if __name__ == '__main__':
    main()