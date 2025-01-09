import os
import torch
import torchaudio
import pandas as pd
import wave
from datasets import load_dataset
from dotenv import load_dotenv
load_dotenv()
SSD_PATH = os.getenv('SSD_PATH')


if __name__ == '__main__':
    dataset = load_dataset('hf-internal-testing/librispeech_asr_dummy', 'clean', split='validation')
    dataset.save_to_disk(f'{SSD_PATH}/librispeech_asr_dummy')
    print(f'The dataset is saved to {SSD_PATH}/librispeech_asr_dummy')
    output_dir = f'{SSD_PATH}/librispeech_asr_dummy/flac/'
    os.makedirs(output_dir, exist_ok=True)

    for idx, sample in enumerate(dataset):
        audio = sample['audio']
        waveform = audio['array']
        sampling_rate = audio['sampling_rate']
        path = audio['path']

        file_path = os.path.join(output_dir, f'{path}')

        torchaudio.save(file_path, torch.tensor([waveform]), sampling_rate)
        try:
            waveform, sr = torchaudio.load(file_path)
            print(f'Saved: {file_path}')
        except Exception as e:
            print(f'Error while reading file {file_path}: {e}')

    print(f'All audio files saved to {output_dir}')
