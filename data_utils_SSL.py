import os
import numpy as np
import torch, glob
import torch.nn as nn
from scipy import signal
from torch import Tensor
import librosa, soundfile
from torch.utils.data import Dataset
import random
import torch.nn.functional as F
import tempfile
from torch.utils.data import Dataset
from torchaudio.io import AudioEffector, CodecConfig
from MaskedSpec import MaskedSpec

def load_data(protocol_path):
    file_list = []
    d_meta ={}

    with open(protocol_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(' ')
            key = parts[1]
            label = parts[-1]
            file_list.append(key)
            
            d_meta[key] = 1 if label == "bonafide" else 0

    return file_list, d_meta

def load_dev_data(protocol_path, base_path):
    file_list=[]
    d_meta = {}
    with open(protocol_path) as f:
        lines = f.readlines()
        for line in lines:  #[:15000]
            parts = line.strip().split(' ')
            key = os.path.join(base_path, parts[1]+'.flac')
            label = parts[-1]
            d_meta[key] = 1 if label == "bonafide" else 0
            file_list.append(key)
            # file_list.append(parts[0])
    return file_list, d_meta             


def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len)+1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x	

	
def trim_silence(wav, top_db=30):
    """
    remove beginning and ending silence
    """
    trimmed_wav, _ = librosa.effects.trim(wav, top_db=top_db)
    return trimmed_wav

 
import numpy as np
import torchaudio

torchaudio.set_audio_backend("sox_io")

def load_audio_robust(path: str, target_sr=16000):
    wav, sr = torchaudio.load(path)          # wav: [C, T]
    wav = wav.mean(dim=0)                    # 下混到单声道 [T]

    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
        sr = target_sr

    return wav.numpy(), sr

class EvalDataset(Dataset):
    def __init__(self, files):
        self.files = files
        self.cut=64600 # take ~4 sec audio (64600 samples)
        self.min_len = 16000
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        utt_id = self.files[idx]
        X,fs = librosa.load(utt_id, sr=16000) 
        X_pad= pad(X,self.cut)
        x_inp= Tensor(X_pad)

        return x_inp,utt_id

class Dataset_ASVspoof5_train(Dataset):
    def __init__(self, args, list_IDs, labels, 
                 database, musan_path=None, rir_path=None, use_musan=True, use_rir=True):
        '''self.list_IDs	: list of strings (each string: utt key),
            self.labels      : dictionary (key: utt key, value: label integer)'''
    
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = database
        self.args=args
        self.MaskedSpec = MaskedSpec()
        self.use_musan = use_musan
        self.use_rir = use_rir
        self.cut=64600 # take ~4 sec audio (64600 samples)

        
        # load MUSAN and RIR 
        if musan_path and use_musan:
            self.noisetypes = ['noise', 'speech', 'music']
            self.noisesnr = {'noise': [0, 15], 'speech': [13, 20], 'music': [5, 15]}
            self.numnoise = {'noise': [1, 1], 'speech': [3, 8], 'music': [1, 1]}
            self.noiselist = {}

            augment_files = glob.glob(os.path.join(musan_path, '*/*/*.wav'))
            for f in augment_files:
                key = f.split('/')[-3]
                if key not in self.noiselist:
                    self.noiselist[key] = []
                self.noiselist[key].append(f)

        if rir_path and use_rir:
            self.rir_files = glob.glob(os.path.join(rir_path, '*/*/*.wav'))
        else:
            self.rir_files = []

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, idx):
        file_name = self.list_IDs[idx]
        label = self.labels[file_name]
        
        audio_wav, sr = load_audio_robust(os.path.join(self.base_dir, file_name+'.flac'), target_sr=16000) 

        # Trim
        # audio_wav = trim_silence(audio_wav, top_db=30)
        
        # RIRS and MUSAN augmentation
        num = random.random()
        if num < 0.5:
            audio_wav = self.apply_augment(audio_wav)
        
        # Padding
        audio_wav = pad(audio_wav, self.cut)
        
        # Masked Spec
        num = random.random()
        if num < 0.5:
            audio_wav = torch.from_numpy(audio_wav).float()
            audio_tensor = self.MaskedSpec.spec_masking(audio_wav, masks_num = 4, f_mask = 30)
        else:
            audio_tensor= torch.FloatTensor(audio_wav)
            
        # audio_tensor= torch.FloatTensor(audio_wav)
        
        return audio_tensor, label
    
    def apply_augment(self, wav):
        if not (self.use_musan or self.use_rir):
            return wav  # 不增强

        # 必须保证二维 [1, T]
        wav = np.stack([wav], axis=0)

        augtype = random.randint(0, 5)
        if augtype == 0:
            pass  # Original
        elif augtype == 1 and self.use_rir:
            wav = self.add_rev(wav)
        elif augtype == 2 and self.use_musan:
            wav = self.add_noise(wav, 'speech')
        elif augtype == 3 and self.use_musan:
            wav = self.add_noise(wav, 'music')
        elif augtype == 4 and self.use_musan:
            wav = self.add_noise(wav, 'noise')
        elif augtype == 5 and self.use_musan:
            wav = self.add_noise(wav, 'speech')
            wav = self.add_noise(wav, 'music')

        return wav[0]

    def add_rev(self, audio):
        rir_file = random.choice(self.rir_files)
        rir, _ = librosa.load(rir_file, sr=None)
        rir = np.expand_dims(rir.astype(np.float64), 0)
        rir = rir / np.sqrt(np.sum(rir ** 2))
        out = signal.convolve(audio, rir, mode='full')[:, :audio.shape[1]]
        return out

    def add_noise(self, audio, noisecat):
        clean_db = 10 * np.log10(np.mean(audio ** 2) + 1e-4)
        numnoise = self.numnoise[noisecat]
        noiselist = random.sample(self.noiselist[noisecat], random.randint(numnoise[0], numnoise[1]))
        noises = []

        length = audio.shape[1]

        for nfile in noiselist:
            noiseaudio, _ = librosa.load(nfile, sr=None)
            if noiseaudio.shape[0] <= length:
                shortage = length - noiseaudio.shape[0]
                noiseaudio = np.pad(noiseaudio, (0, shortage), 'wrap')

            start = int(random.random() * (noiseaudio.shape[0] - length))
            noiseaudio = noiseaudio[start:start + length]
            noiseaudio = np.stack([noiseaudio], axis=0)

            noise_db = 10 * np.log10(np.mean(noiseaudio ** 2) + 1e-4)
            noisesnr = random.uniform(self.noisesnr[noisecat][0], self.noisesnr[noisecat][1])
            noises.append(np.sqrt(10 ** ((clean_db - noise_db - noisesnr) / 10)) * noiseaudio)

        noise = np.sum(np.concatenate(noises, axis=0), axis=0, keepdims=True)
        return noise + audio


class EvalDataset(Dataset):
    def __init__(self, files, labels):
        self.files = files
        self.labels = labels
        self.cut=64600 # take ~4 sec audio (64600 samples)
        self.min_len = 16000
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        utt_id = self.files[idx]
        label = self.labels[utt_id]
        X,fs = librosa.load(utt_id, sr=16000) 
        X_pad= pad(X,self.cut)
        x_inp= Tensor(X_pad)

        return x_inp,label


