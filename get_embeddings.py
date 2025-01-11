import os
import torch
import numpy as np


def process_with_hubert(waveform, sampling_rate, processor, model, console=None):
    input_values = processor(
        waveform,
        sampling_rate=sampling_rate,
        return_tensors='pt',
    ).input_values

    with torch.no_grad():
        outputs = model(input_values)

    embeddings = outputs.last_hidden_state.numpy()
    # console.print(f'Embedding shape: {embeddings.shape}', style='bold blue')
    if console:
        console.print(f'Number of frames: {embeddings.shape[1]}', style='bold green')
    return embeddings


def save_embedding(embedding, output_directory, filename, console):
    output_file_path = os.path.join(output_directory, filename)
    np.save(output_file_path, embedding)
    if console:
        console.print(f'Saved embedding to: {output_file_path}', style='bold red')
