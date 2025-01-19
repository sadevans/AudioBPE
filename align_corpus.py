import os
import os.path as osp
import shutil
import argparse


def main(root_wav_dir, root_txt_dir, root_corpus_dir, root_textgrid_dir):
    assert osp.exists(root_wav_dir), f'{root_wav_dir = }'
    assert osp.exists(root_txt_dir), f'{root_txt_dir = }'
    os.makedirs(root_corpus_dir, exist_ok=True)

    for wav_file in os.listdir(root_wav_dir):
        if wav_file.find('wav') == -1:
            continue
        wav_path = osp.join(root_wav_dir, wav_file)
        new_wav_path = osp.join(root_corpus_dir, wav_file)
        shutil.copy(wav_path, new_wav_path)
        txt_path = osp.join(root_txt_dir, wav_file.replace('wav', 'txt'))
        new_txt_path = new_wav_path.replace('wav', 'txt')
        shutil.copy(txt_path, new_txt_path)

    os.system(f'mfa align {root_corpus_dir} english_us_arpa english_us_arpa {root_textgrid_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process and align audio files for MFA.')
    parser.add_argument('--root_wav_dir', type=str, required=True, help='Path to the root wav directory.')
    parser.add_argument('--root_txt_dir', type=str, required=True, help='Path to the root txt directory.')
    parser.add_argument('--root_corpus_dir', type=str, required=True, help='Path to the root corpus directory.')
    parser.add_argument('--root_textgrid_dir', type=str, required=True, help='Path to the root aligned TextGrid directory.')

    args = parser.parse_args()
    main(args.root_wav_dir, args.root_txt_dir, args.root_corpus_dir, args.root_textgrid_dir)