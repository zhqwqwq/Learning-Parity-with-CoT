

# Aprendizaje de Paridad con Chain-of-Thought

Este repositorio contiene el código para reproducir los resultados de nuestro artículo *From Sparse Dependence to Sparse Attention: Unveiling How Chain-of-Thought Enhances Transformer Sample Efficiency*.

## Prerequisitos

* [PyTorch](https://pytorch.org/get-started/locally/)
* [transformers](https://github.com/huggingface/transformers) 

## Generación de Datos

Para generar datos de paridad sintéticos, primero crea un archivo de configuración `data_config.py`. A continuación se muestra un ejemplo de configuración (`configs/data_config.py`): 

```python
{
    'dataset_type': 'BinaryDataset',
    'n_samples': 10000,
    'val_samples': 2048,
    'n_digit': 30,
    'n_secret': 3,
    'use_cot': True,
}
```

- `n_samples`: Número de muestras de entrenamiento.
- `val_samples`: Número de muestras de validación.
- `n_digit`: Número de variables de entrada.
- `n_secret`: Número de variables secretas.
- `use_cot`: Establece en `True` para usar datos de Chain-of-Thought (CoT).

Después de definir la configuración, genera los datos ejecutando:

```bash
python src/generate_data.py --config path-to-data_config.py
```

## Entrenamiento

Para entrenar un modelo Transformer en el problema de paridad, utiliza el siguiente comando:

```bash
export CUDA_VISIBLE_DEVICES=0
PROJECT=Istree
MODEL=transformer
total_training_sample=100000
training_samples=10000
n_digits=30
k=3
CoT=True
lr=6e-5
num_layers=4
num_heads=3
python src/train.py \
    --world_size 1 \
    --total_training_samples ${training_samples} \
    --model_type transformer \
    --model_config_path config/gpt2_tiny_wpetrain.py \
    --dataset_dir data/Nonintersect_Binary/binary_${training_samples}_${n_digits}_${k}_${CoT}_False_False \
    --dataset_type BinaryDataset \
    --output_dir model/ \
    --batch_size 512 \
    --lr ${lr} \
    --weight_decay 0 \
    --log_interval 2048 \
    --save_interval 2048 \
    --eval_interval 2048 \
    --report_to_wandb \
   --num_hidden_layers ${num_layers} \
   --num_attention_heads ${num_heads} \
```

Donde:

- `training_samples`: Número de muestras en el conjunto de entrenamiento.
- `total_training_samples`: Número total de muestras utilizadas durante el entrenamiento (`iteraciones = total_training_samples / batch_size`).
- `lr`: Tasa de aprendizaje.
- `num_layers`: Número de capas ocultas en el modelo.
- `num_heads`: Número de cabezales de atención.
- `CoT`: Establece en `True` para entrenar con datos de Chain-of-Thought, o `False` de lo contrario.

Los resultados del entrenamiento se guardarán en el directorio `model/`.

## Reproducción de los Resultados

### Figura 1, 2

Para reproducir los resultados mostrados en las Figuras 1 y 2 de nuestro artículo, sigue estos pasos:

1. Utiliza las siguientes configuraciones para generar datos:

| n_samples  | n_digits | n_secret  | use_cot     |
| ---------- | -------- | --------- | ----------- |
| $10000000$ | $30$     | $1,2,3,4$ | True, False |

2. Entrena el modelo con la siguiente configuración:

| total_training_samples | training_samples | n_digits | k         | CoT         | num_layers | num_heads | lr                                              |
| ---------------------- | ---------------- | -------- | --------- | ----------- | ---------- | --------- | ----------------------------------------------- |
| $10^7$                 | $10^7$           | $30$     | $1,2,3,4$ | True, False | $1,2,3,4$  | $1,2,3,4$ | $6\times10^{-5}, 8\times10^{-5},1\times10^{-4}$ |

3. Para reproducir la Figura 1, ejecuta `Figures/Fig1/fig1.1.py` y `Figures/Fig1/fig1.2.py`.

   Para reproducir la Figura 2, ejecuta `Figures/Fig2/fig2.1.py`, `Figures/Fig2/fig2.2.py` y `Figures/Fig2/fig2.3.py`.

### Figura 3, 8

1. Utiliza las siguientes configuraciones para generar datos:

| n_samples  | n_digits | n_secret           | use_cot |
| ---------- | -------- | ------------------ | ------- |
| $1000000$  | $100$    | $20,30,\cdots,100$ | True    |
| $10000000$ | $40$     | $20$               | True    |

2. Entrena el modelo con la siguiente configuración:

| total_training_samples | training_samples | n_digits | k                  | CoT  | num_layers | num_heads | lr                                              |
| ---------------------- | ---------------- | -------- | ------------------ | ---- | ---------- | --------- | ----------------------------------------------- |
| $10^7$                 | $10^6$           | $100$    | $20,30,\cdots,100$ | True | $1$        | $1$       | $6\times10^{-5}, 8\times10^{-5},1\times10^{-4}$ |
| $10^7$                 | $10^7$           | $40$     | $20$               | True | $1$        | $1$       | $6\times10^{-5}$                                |
| $10^7$                 | $10^7$           | $40$     | $20$               | True | $2$        | $2$       | $6\times10^{-5}$                                |

3. Para reproducir la Figura 3, ejecuta `Figures/Fig2/fig3.1.py` y`Figures/Fig2/fig3.2.py`.

### Figura 4

1. Utiliza las siguientes configuraciones para generar datos:

| n_samples | n_digits | n_secret | use_cot     |
| --------- | -------- | -------- | ----------- |
| $10000$   | $20$     | $6$      | True, False |

2. Entrena el modelo con la siguiente configuración:

| total_training_samples | training_samples | n_digits | k    | CoT   | num_layers | num_heads | lr                |
| ---------------------- | ---------------- | -------- | ---- | ----- | ---------- | --------- | ----------------- |
| $10^7$                 | $10000$          | $20$     | $6$  | False | $1,2,3,4$  | $1,2,3,4$ | $1\times10^{-4}$  |
| $10^7$                 | $10000$          | $20$     | $6$  | True  | $1$        | $1$       | $1\times 10^{-5}$ |

3. Para reproducir la Figura 4, ejecuta `Figures/Fig4/fig4.py`.

### Figura 5, 10, 11

1. Utiliza las siguientes configuraciones para generar datos:

| n_samples                    | n_digits | n_secret | use_cot |
| ---------------------------- | -------- | -------- | ------- |
| $5000,10000,50000,10^5,10^6$ | $20$     | $6$      | False   |

2. Entrena el modelo con la siguiente configuración:

| total_training_samples | training_samples             | n_digits | k    | CoT   | num_layers | num_heads | lr               |
| ---------------------- | ---------------------------- | -------- | ---- | ----- | ---------- | --------- | ---------------- |
| $10^7$                 | $5000,10000,50000,10^5,10^6$ | $20$     | $6$  | False | $1$        | $2$       | $1\times10^{-4}$ |
| $10^7$                 | $5000,10000,50000,10^5,10^6$ | $20$     | $6$  | False | $2$        | $3$       | $1\times10^{-4}$ |
| $10^7$                 | $5000,10000,50000,10^5,10^6$ | $20$     | $6$  | False | $4$        | $4$       | $1\times10^{-4}$ |

3. Para reproducir la Figura 5, ejecuta `Figures/Fig5/fig5.1.py`, `Figures/Fig5/fig5.2.py`.

   Para reproducir la Figura 10, ejecuta `Figures/Fig10/fig10.py`

   Para reproducir la Figura 11, ejecuta `Figures/Fig11/fig11.py`

### Figura 9

1. Utiliza las siguientes configuraciones para generar datos:

| n_samples        | n_digits | n_secret | use_cot |
| ---------------- | -------- | -------- | ------- |
| $10^4,10^5,10^6$ | $20$     | $12$     | False   |
| $10^4,10^6$      | $20$     | $12$     | True    |

2. Entrena el modelo con la siguiente configuración:

| total_training_samples | training_samples | n_digits | k    | CoT  | num_layers    | num_heads     | lr                                               |
| ---------------------- | ---------------- | -------- | ---- | ---- | ------------- | ------------- | ------------------------------------------------ |
| $10^7$                 | $10^4,10^5,10^6$ | $20$     | $12$ | True | $1,2,3,4,6,8$ | $1,2,3,4,6,8$ | $6\times10^{-5}, 8\times10^{-5},1\times10^{-4}$  |
| $10^7$                 | $10^4,10^6$      | $20$     | $12$ | True | $1$           | $1$           | $6\times10^{-5},8\times 10 ^{-5},1\times10^{-4}$ |

3. Para reproducir la Figura 9, ejecuta `Figures/Fig9/fig9.py`.

### Figura 6,7

Para reproducir las Figuras 6 y 7, primero ejecuta `Figures/Fig6-7/work.py` para calcular la entropía de atención normalizada. Luego ejecuta `Figures/Fig6-7/fig6.py` y `Figures/Fig6-7/fig7.py`.
