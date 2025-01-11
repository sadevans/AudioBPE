import os
import glob
import torch
import numpy as np
import soundfile as sf
from rich.console import Console
from audio import AudioWaveform
from tokenizer import AdaptiveAudioAmplitudeTokenizer
from get_embeddings import process_with_hubert, save_embedding
from transformers import AutoProcessor, HubertModel
from dotenv import load_dotenv

load_dotenv()
SSD_PATH = os.getenv('SSD_PATH')
DATASET_NAME = os.getenv('DATASET_NAME')

console = Console(record=True)
processor = AutoProcessor.from_pretrained('facebook/hubert-large-ls960-ft')
model = HubertModel.from_pretrained('facebook/hubert-large-ls960-ft')


def process_flac_file(input_file_path, output_directory, emb_output_directory):
    console.print('------------------------------------')

    id_ = input_file_path.split('/')[-1].split('.flac')[0]
    waveform, sampling_rate = sf.read(input_file_path, dtype='float32')
    console.print(f'Sampling rate: {sampling_rate}')

    if len(waveform.shape) > 1:
        waveform = waveform[:, 0]
    console.print(f'Waveform shape: {waveform.shape}')

    audio_waveform_sr = AudioWaveform(waveform, sampling_rate)

    tokenizer = AdaptiveAudioAmplitudeTokenizer(sampling_rate=sampling_rate)

    segments, _ = tokenizer.tokenize(audio_waveform_sr)
    console.print(f'Number of segments: {len(segments)}', style='bold yellow')

    os.makedirs(os.path.join(output_directory, id_), exist_ok=True)
    os.makedirs(os.path.join(emb_output_directory, id_), exist_ok=True)

    for idx, segment in enumerate(segments):
        output_file_path = os.path.join(output_directory, id_, f'{id_}_{idx + 1}.flac')
        sf.write(output_file_path, segment.waveform, segment.sampling_rate)
        console.print(f'Saved segment {idx + 1} to: {output_file_path}', style='bold green')

        hubert_embedding = process_with_hubert(
            segment.waveform,
            segment.sampling_rate,
            processor,
            model,
            console=console,
            )

        save_embedding(
            hubert_embedding,
            os.path.join(emb_output_directory, id_),
            f'{id_}_{idx + 1}.npy',
            console=console,
        )


if __name__ == '__main__':
    flac_files = glob.glob(f'{SSD_PATH}/{DATASET_NAME}/flac/*.flac')
    OUTPUT_DIR = f'{SSD_PATH}/{DATASET_NAME}/amplitude_tokenized/'
    EMB_OUTPUT_DIR = f'{SSD_PATH}/{DATASET_NAME}/hubert_embeddings/'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(EMB_OUTPUT_DIR, exist_ok=True)

    for input_flac_file in flac_files:
        process_flac_file(input_flac_file, OUTPUT_DIR, EMB_OUTPUT_DIR)
