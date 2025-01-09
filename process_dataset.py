import os
import glob
import soundfile as sf
from audio import AudioWaveform
from tokenizer import AdaptiveAudioAmplitudeTokenizer
from dotenv import load_dotenv

load_dotenv()
SSD_PATH = os.getenv('SSD_PATH')
DATASET_NAME = os.getenv('DATASET_NAME')


def process_flac_file(input_file_path, output_directory):
    id_ = input_file_path.split('/')[-1].split('.flac')[0]
    waveform, sampling_rate = sf.read(input_file_path, dtype='float32')
    print(f'Sampling rate: {sampling_rate}')

    if len(waveform.shape) > 1:
        waveform = waveform[:, 0]
    print(f'Waveform shape: {waveform.shape}')

    audio_waveform_sr = AudioWaveform(waveform, sampling_rate)

    tokenizer = AdaptiveAudioAmplitudeTokenizer(sampling_rate=sampling_rate)

    segments, _ = tokenizer.tokenize(audio_waveform_sr)
    print(f'Number of segments: {len(segments)}')

    for idx, segment in enumerate(segments):
        output_file_path = os.path.join(output_directory, f'{id_}_{idx + 1}.flac')
        sf.write(output_file_path, segment.waveform, segment.sampling_rate)
        print(f'Saved segment {idx + 1} to: {output_file_path}')


if __name__ == '__main__':
    flac_files = glob.glob(f'{SSD_PATH}/{DATASET_NAME}/flac/*.flac')
    OUTPUT_DIR = f'{SSD_PATH}/{DATASET_NAME}/amplitude_tokenized/'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for input_flac_file in flac_files:
        process_flac_file(input_flac_file, OUTPUT_DIR)
