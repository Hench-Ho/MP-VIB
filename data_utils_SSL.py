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

def load_data(protocol_path, spk_col=0, atk_col=None):
    file_list = []
    d_meta = {}

    spk_id = {}
    attack_id = {}

    spk2idx = {}
    atk2idx = {}

    def _get_or_add(mapping, name):
        if name not in mapping:
            mapping[name] = len(mapping)
        return mapping[name]

    def _infer_attack(parts):
        candidates = []
        candidates.append(parts[-3])

        # 也可能是某个固定字段，如 LA/PA, codec 等；这里尽量不乱猜：
        for p in parts:
            # 非常粗的过滤：跳过明显不是 attack 的字段
            if p in ("bonafide", "spoof"):
                continue
            if p.startswith("utt") or p.endswith(".wav"):
                continue
            # 有些协议会写 "A01" "A02" 或 "CC1" 等
            if (len(p) <= 10) and (any(c.isdigit() for c in p)) and (any(c.isalpha() for c in p)):
                candidates.append(p)

        # 去重并选一个最合理的
        for c in candidates:
            if c and c not in ("-", "na", "N/A"):
                return c
        return "unknown"

    with open(protocol_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(' ')
            key = parts[1]
            label = parts[-1]
            if label not in ["bonafide", "spoof"]:
                raise ValueError(f"label error: {label} in line: {line}")

            file_list.append(key)
            d_meta[key] = 1 if label == "bonafide" else 0

            # -------- speaker --------
            spk_name = parts[0]
            spk_id[key] = _get_or_add(spk2idx, spk_name)

            # -------- attack --------
            atk_name = parts[-2]
            attack_id[key] = _get_or_add(atk2idx, atk_name)

    return file_list, d_meta, spk_id, attack_id, spk2idx, atk2idx

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
    def __init__(self, args, list_IDs, labels, spk_id, attack_id,
                 database, musan_path=None, rir_path=None, use_musan=True, use_rir=True):
        '''self.list_IDs	: list of strings (each string: utt key),
            self.labels      : dictionary (key: utt key, value: label integer)'''
    
        self.list_IDs = list_IDs
        self.labels = labels
        self.spk_id = spk_id
        self.attack_id = attack_id
        self.base_dir = database
        self.args=args
        self.codec_aug = RandomCodecAug(p=0.5)
        self.comp_aug = CompressionAugment(p=0.4, mp3_kbps=16, m4a_kbps=64)
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
        spk   = self.spk_id[file_name]
        atk   = self.attack_id[file_name]
        
        audio_wav, sr = load_audio_robust(os.path.join(self.base_dir, file_name+'.flac'), target_sr=16000) 

        # Trim
        # audio_wav = trim_silence(audio_wav, top_db=30)
        
        # RIRS and MUSAN augmentation
        # num = random.random()
        # if num < 0.5:
        #     audio_wav = self.apply_augment(audio_wav)
        
        # # codec
        # num = random.random()
        # if num < 0.5:
        #     audio_wav = np.asarray(audio_wav, dtype=np.float32)
        #     audio_wav = self.codec_aug(audio_wav, sr)
        
        # # Compose
        # num = random.random()
        # if num < 0.5:
        #     audio_wav = self.comp_aug(audio_wav, sr)
        #     audio_wav = np.asarray(audio_wav, dtype=np.float32)
        
        # Padding
        audio_wav = pad(audio_wav, self.cut)
        
        # # Masked Spec
        # num = random.random()
        # if num < 0.5:
        #     audio_wav = torch.from_numpy(audio_wav).float()
        #     audio_tensor = self.MaskedSpec.spec_masking(audio_wav, masks_num = 4, f_mask = 30)
        # else:
        #     audio_tensor= torch.FloatTensor(audio_wav)
            
        # soundfile.write('debug_fake.wav', audio_wav, sr)
        audio_tensor= torch.FloatTensor(audio_wav)
        return audio_tensor, label, spk, atk
    
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


def _ensure_2d_torch(wav: torch.Tensor) -> torch.Tensor:
    # [T] -> [1, T]; [C, T] keep
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    return wav

def _to_numpy_1d(x):
    if isinstance(x, np.ndarray):
        return x
    x = x.detach().cpu()
    if x.dim() == 2 and x.size(0) == 1:
        x = x.squeeze(0)
    return x.numpy()

def compress_decompress_torchaudio(
    wav,                      # np.ndarray [T] or torch.Tensor [T]/[C,T]
    sr: int,
    kind: str,                # "mp3" or "m4a"
    bitrate_kbps: int,
):
    """
    return type matches input type (numpy in -> numpy out, torch in -> torch out)
    """
    import torchaudio
    from torchaudio.io import StreamWriter

    is_numpy = isinstance(wav, np.ndarray)
    if is_numpy:
        wav = torch.from_numpy(wav)

    wav = wav.detach().cpu().float()
    wav = _ensure_2d_torch(wav)                 # [C, T]
    C, T = wav.shape
    wav_TC = wav.transpose(0, 1).contiguous()   # [T, C]

    if kind == "mp3":
        suffix = ".mp3"
        container_format = "mp3"
        codec = "libmp3lame"
        bitrate = int(bitrate_kbps * 1000)
    elif kind == "m4a":
        suffix = ".m4a"
        container_format = "ipod"   # 生成 m4a 常用；不行可改 "mp4"
        codec = "aac"
        bitrate = int(bitrate_kbps * 1000)
    else:
        raise ValueError(f"Unknown kind: {kind}")

    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp_path = tmp.name
    tmp.close()

    try:
        writer = StreamWriter(tmp_path, format=container_format)
        writer.add_audio_stream(
            sample_rate=sr,
            num_channels=C,
            codec=codec,
            bitrate=bitrate,
        )
        with writer.open():
            writer.write_audio_chunk(0, wav_TC)

        wav2, sr2 = torchaudio.load(tmp_path)   # [C, T2]
        if sr2 != sr:
            wav2 = torchaudio.functional.resample(wav2, sr2, sr)

        # match length
        if wav2.size(1) > T:
            wav2 = wav2[:, :T]
        elif wav2.size(1) < T:
            wav2 = F.pad(wav2, (0, T - wav2.size(1)))

        return _to_numpy_1d(wav2) if is_numpy else wav2

    except Exception as e:
        # ✅ 编码/解码失败（ffmpeg/codec 缺失等）就直接回退原始 wav，不让训练崩
        return _to_numpy_1d(wav) if is_numpy else wav

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

class CompressionAugment:
    def __init__(self, p=0.5, mp3_kbps=16, m4a_kbps=64):
        self.p = float(p)
        self.mp3_kbps = int(mp3_kbps)
        self.m4a_kbps = int(m4a_kbps)

    def __call__(self, wav, sr: int):
        if random.random() > self.p:
            return wav
        if random.random() < 0.5:
            return compress_decompress_torchaudio(wav, sr, "mp3", self.mp3_kbps)
        else:
            return compress_decompress_torchaudio(wav, sr, "m4a", self.m4a_kbps)



class RandomCodecAug:
    """
    随机选 codec 做编码-解码增强（MP3 / OGG-OPUS / OGG-VORBIS / WEBM-OPUS）
    """
    def __init__(self, p=0.5):
        self.p = float(p)

    def _make_effector(self):
        choice = random.choice(["mp3_vbr", "mp3_cbr", "ogg_opus", "ogg_vorbis", "webm_opus"])

        if choice == "mp3_vbr":
            q = random.randint(2, 8)
            return AudioEffector(format="mp3", codec_config=CodecConfig(qscale=q), pad_end=True)

        if choice == "mp3_cbr":
            br = random.choice([16000, 24000, 32000, 64000])
            return AudioEffector(format="mp3", codec_config=CodecConfig(bit_rate=br), pad_end=True)

        if choice == "ogg_opus":
            # 优先 libopus，失败回退 opus
            try:
                return AudioEffector(format="ogg", encoder="libopus", pad_end=True)
            except Exception:
                return AudioEffector(format="ogg", encoder="opus", pad_end=True)

        if choice == "ogg_vorbis":
            # 优先 libvorbis，失败回退 vorbis
            try:
                return AudioEffector(format="ogg", encoder="libvorbis", pad_end=True)
            except Exception:
                return AudioEffector(format="ogg", encoder="vorbis", pad_end=True)

        # webm + opus：优先 libopus
        try:
            return AudioEffector(format="webm", encoder="libopus", pad_end=True)
        except Exception:
            return AudioEffector(format="webm", encoder="opus", pad_end=True)

    @torch.no_grad()
    def __call__(self, waveform, sample_rate: int):
        import numpy as np
        import torch
        import torch.nn.functional as F

        is_numpy = isinstance(waveform, np.ndarray)
        if is_numpy:
            waveform = torch.from_numpy(waveform)

        if waveform.dtype == torch.float64:
            waveform = waveform.float()

        # 记录原始是否 1D（你的主流程就是 1D）
        orig_1d = (waveform.dim() == 1)

        if torch.rand(()) > self.p:
            out = waveform
            if orig_1d and out.dim() == 2 and out.shape[0] == 1:
                out = out.squeeze(0)
            return out.numpy() if is_numpy else out

        # 统一成 [1, T]
        if orig_1d:
            waveform = waveform.unsqueeze(0)

        if waveform.dim() != 2:
            raise ValueError("waveform must be 1D or 2D.")

        # 喂给 AudioEffector: (time, channel)
        # 你这里 waveform 是 [1, T]，所以转成 [T, 1]
        x = waveform.t().contiguous()
        T0 = x.shape[0]

        eff = self._make_effector()
        y = eff.apply(x.cpu(), sample_rate)  # [T,1]

        # pad/crop 回原长度
        if y.shape[0] < T0:
            y = F.pad(y, (0, 0, 0, T0 - y.shape[0]))
        else:
            y = y[:T0, :]

        out = y.t()  # [1,T]

        # ✅ 返回和输入一致：输入是 1D，就 squeeze 回 1D
        if orig_1d:
            out = out.squeeze(0)

        return out.numpy() if is_numpy else out



