import numpy as np
import subprocess
from iat import iat
import sys
import torch
from transformers import BertTokenizer, BertModel, Wav2Vec2Model, Wav2Vec2Processor
from moviepy.editor import *
import os
import torchaudio
from torchaudio import transforms


def resample(in_path):
    out_path = in_path[:-4] + '_resampled.mp3'
    if os.path.exists(out_path):
        return out_path
    waveform, sr = torchaudio.load(in_path)
    transform = transforms.Resample(sr, 16000)
    waveform = transform(waveform)
    torchaudio.save(out_path, waveform, sr)
    return out_path


def video2audio(in_path):
    out_path = in_path[:-4] + '.mp3'
    if os.path.exists(out_path):
        return out_path
    video = VideoFileClip(in_path)
    video.audio.write_audiofile(out_path)
    return out_path


def video_feature_extraction(in_paths, gluon_cv_path='gluon-cv/scripts/action-recognition/feat_extract.py'):
    data_dir = os.path.dirname(in_paths[0])

    def out_str(in_str):
        return os.path.join(os.path.dirname(in_str),
                            f'r2plus1d_resnet50_kinetics400_{os.path.basename(in_str)}_feat.npy')

    out_paths = [out_str(in_path) for in_path in in_paths]

    tmp_file_path = os.path.join(data_dir, 'tmp_video.txt')
    paths = [p for p in in_paths if not os.path.exists(out_str(p))]
    np.savetxt(tmp_file_path, paths, delimiter='\n', fmt='%s')
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = "/usr/local/cuda-10.0/lib64:/usr/local/cuda/lib64:"
    subprocess.call([sys.executable, gluon_cv_path, '--data-list', tmp_file_path, '--model',
                     'r2plus1d_resnet50_kinetics400', '--save-dir', data_dir], env=env)
    return out_paths


# 从音频中提取文本
def asr(in_path):
    out_path = in_path[:-len('_resampled.mp3')] + '.txt'
    if os.path.exists(out_path):
        return out_path
    iat(in_path, out_path)
    return out_path


def bert(model, tk, s):
    encoded_t = tk(s, max_length=100, padding='max_length', truncation=True, return_tensors='pt')
    with torch.no_grad():
        vec = model(**encoded_t).last_hidden_state.squeeze(0)
    return vec


def text_feature_extraction(in_paths, bert_path='bert'):
    out_paths = []
    model = BertModel.from_pretrained(bert_path)
    tokenizer = BertTokenizer.from_pretrained(bert_path)
    for in_path in in_paths:
        out_path = in_path[:-4] + '_live_text_vec'
        if os.path.exists(out_path + '.npy'):
            out_paths.append(out_path + '.npy')
            continue
        with open(in_path, 'r') as in_f:
            lines = in_f.read()
        vec = bert(model, tokenizer, lines).numpy()
        np.save(out_path, vec)
        out_paths.append(out_path + '.npy')
    return out_paths


def script_feature_extraction(in_paths, bert_path='bert'):
    out_paths = []
    model = BertModel.from_pretrained(bert_path)
    tokenizer = BertTokenizer.from_pretrained(bert_path)
    for in_path in in_paths:
        out_path = in_path[:-4] + '_script_text_vec'
        if os.path.exists(out_path + '.npy'):
            out_paths.append(out_path + '.npy')
            continue

        with open(in_path, 'r') as in_f:
            lines = in_f.read()
        vec = bert(model, tokenizer, lines).numpy()
        np.save(out_path, vec)
        out_paths.append(out_path + '.npy')
    return out_paths


def audio_feature_extraction(in_paths, target_length=16000 * 12):
    out_paths = []
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
    for in_path in in_paths:
        out_path = in_path[:-4] + '_audio_vec'
        if os.path.exists(out_path + '.npy'):
            out_paths.append(out_path + '.npy')
            continue
        waveform, sr = torchaudio.load(in_path)
        waveform = waveform[0]
        p = target_length - len(waveform)
        if p > 0:
            waveform = torch.cat([waveform, torch.zeros(p)], dim=-1)
        elif p < 0:
            waveform = waveform[:target_length]
        assert waveform.shape == (target_length,)
        inputs = processor(waveform, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)

        vec = outputs.extract_features.squeeze(0).numpy()
        np.save(out_path, vec)
        out_paths.append(out_path + '.npy')
    return out_paths


def prepare_feature_file(args):
    v_paths = args['input_videos']
    s_paths = args['input_scripts']
    # separate audio from video clip
    mid_paths = [video2audio(p) for p in v_paths]
    # resample audio before asr
    a_paths = [resample(p) for p in mid_paths]
    # get text from audio
    lt_paths = [asr(p) for p in a_paths]

    lt_vec_paths = text_feature_extraction(lt_paths)
    a_vec_paths = audio_feature_extraction(a_paths)
    v_vec_paths = video_feature_extraction(v_paths)
    s_vec_paths = script_feature_extraction(s_paths)

    return lt_vec_paths, a_vec_paths, v_vec_paths, s_vec_paths


if __name__ == '__main__':
    args = {
        'input_videos': ['data/sample.mp4'],
        'input_scripts': ['data/sample.script'],
    }
    prepare_feature_file(args)
