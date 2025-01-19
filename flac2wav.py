import os
import argparse
from pydub import AudioSegment


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process and align audio files for MFA.')
    parser.add_argument('--flac_dir', type=str, required=True, help='Path to the root flac directory.')
    parser.add_argument('--wav_dir', type=str, required=True, help='Path to the root wav directory.')

    args = parser.parse_args()

    input_dir = args.flac_dir
    output_dir = args.wav_dir
    os.makedirs(output_dir, exist_ok=True)

    for file_name in os.listdir(input_dir):
        if file_name.endswith('.flac'):
            flac_path = os.path.join(input_dir, file_name)
            wav_path = os.path.join(output_dir, file_name.replace('.flac', '.wav'))

            audio = AudioSegment.from_file(flac_path, format='flac')
            audio = audio.set_channels(1).set_frame_rate(16000)
            audio.export(wav_path, format='wav')
            print(f'Converted: {file_name} -> {wav_path}')
