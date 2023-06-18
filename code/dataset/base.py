import random
import os
import csv
import numpy as np
import torch
import torch.utils.data as torchdata
from torchvision import transforms
import torchaudio
import librosa
from PIL import Image
import torch.nn.functional as F

from . import video_transforms as vtransforms

class BaseAVEDataset(torchdata.Dataset):
    def __init__(self, list_sample, opt, max_sample=-1, split='train'):
        # params
        self.num_frames = opt.num_frames
        self.stride_frames = opt.stride_frames
        self.frameRate = opt.frameRate
        self.imgSize = opt.imgSize
        self.audRate = opt.audRate
        self.audLen = opt.audLen
        self.audSec = 1. * self.audLen / self.audRate
        self.binary_mask = opt.binary_mask

        # STFT params
        self.log_freq = opt.log_freq
        self.stft_frame = opt.stft_frame
        self.stft_hop = opt.stft_hop
        self.HS = opt.stft_frame // 2 + 1
        self.WS = (self.audLen + 1) // self.stft_hop

        self.split = split
        self.seed = opt.seed
        random.seed(self.seed)

        # initialize video transform
        self._init_vtransform()

        # list_sample can be a python list or a csv file of list
        if isinstance(list_sample, str):
            self.list_sample = []
            for row in csv.reader(open(list_sample, 'r'), delimiter=','):
                if len(row) < 2:
                    continue
                self.list_sample.append(row)
        elif isinstance(list_sample, list):
            self.list_sample = list_sample
        else:
            raise('Error list_sample!')

        if self.split == 'train':
            self.list_sample *= opt.dup_trainset
            random.shuffle(self.list_sample)

        if max_sample > 0:
            self.list_sample = self.list_sample[0:max_sample]

        num_sample = len(self.list_sample)
        assert num_sample > 0
        print('# samples: {}'.format(num_sample))

    def __len__(self):
        return len(self.list_sample)

    # video transform funcs
    def _init_vtransform(self):
        transform_list = []
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        if self.split == 'train':
            transform_list.append(vtransforms.Resize(int(self.imgSize * 1.1), Image.BICUBIC))
            transform_list.append(vtransforms.RandomCrop(self.imgSize))
            transform_list.append(vtransforms.RandomHorizontalFlip())
        else:
            transform_list.append(vtransforms.Resize(self.imgSize, Image.BICUBIC))
            transform_list.append(vtransforms.CenterCrop(self.imgSize))

        transform_list.append(vtransforms.ToTensor())
        transform_list.append(vtransforms.Normalize(mean, std))
        transform_list.append(vtransforms.Stack())
        self.vid_transform = transforms.Compose(transform_list)

    # image transform funcs, deprecated
    def _init_transform(self):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        if self.split == 'train':
            self.img_transform = transforms.Compose([
                transforms.Scale(int(self.imgSize)),
                #transforms.RandomCrop(self.imgSize),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])
        else:
            self.img_transform = transforms.Compose([
                transforms.Scale(self.imgSize),
                transforms.CenterCrop(self.imgSize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])
            
    def _load_frames_21(self, paths):
        frames = []
        for path in paths:
            frames.append(self._load_frame_21(path))
        frames = self.vid_transform(frames)
        return frames
    
    def _load_frames_ave(self, paths):
        frames = []
        for path in paths:
            frames.append(self._load_frame_ave(path))
        frames = self.vid_transform(frames)

    def _load_frames(self, paths):
        frames = []
        for path in paths:
            frames.append(self._load_frame(path))
        frames = self.vid_transform(frames)
        return frames

    def _load_frames_det(self, paths, path_frames_ids,  path_frames_det):
        det_res = np.load(path_frames_det)
        frames = []
        N = len(paths)
        for n in range(N):
            path = paths[n]
            id = path_frames_ids[n]
            frames.append(self._load_frame_det(path, id, det_res))
        frames = self.vid_transform(frames)
        return frames
    def _load_frames_det_21(self, paths, path_frames_ids,  path_frames_det, current_cls):
        det_res = np.load(path_frames_det)
        frames = []
        N = len(paths)
        for n in range(N):
            path = paths[n]
            id = path_frames_ids[n]
            frames.append(self._load_frame_det_21(path, id, det_res, current_cls))
        frames = self.vid_transform(frames)
        return frames
    def _load_frames_det_ave(self, paths, path_frames_ids, path_frames_det, current_cls):
        det_res = np.load(path_frames_det)
        frames = []
        N = len(paths)
        for n in range(N):
            path = paths[n]
            id = path_frames_ids[n]
            frames.append(self._load_frame_det_ave(path, id, det_res, current_cls))
        frames = self.vid_transform(frames)
        return frames
    #pooling alongside the temporal dimension: C, T, H, W -> C, H, W
    def temporal_pooling(self, x):
        C, T, H, W = x.shape
        x = x.permute(0,2,3,1)
        x = x.view(C,H*W,T)
        x = F.adaptive_max_pool1d(x, 1)
        x = x.view(C, H, W)
        return x
    #motion loading

    #C, T, H, W -> C, T spatial pooling
    def spatial_pooling(self, x):
        C, T, H, W = x.shape
        x = F.adaptive_max_pool2d(x, 1)
        x = x.view(C, T)
        return x
    
    def _load_motion_tensor_time(self, path):
        motion = np.load(path)
        x = torch.tensor(motion)
        x = self.spatial_pooling(x)
        return x
    
    def _load_motion_tensor(self, path):
        motion = np.load(path)
        x = torch.tensor(motion)
        x = self.temporal_pooling(x)
        return x

    def _load_frame_det(self, path, id, det_res):

        # load image
        img = Image.open(path).convert('RGB')

        # get box
        idx = np.where(det_res[:, 0] == id)
        if len(idx[0])!=0:
            n = np.argmax(det_res[idx, 2], axis=1)
            bb = det_res[idx[0][n[0]], 3:]
            # crop image
            img = img.crop((bb[0], bb[1], bb[2], bb[3]))
        return img
    
    def _load_frame_det_21(self, path, id, det_res, cur_cls):
        # load image
        img = Image.open(path).convert('RGB')

        # get box
        class_id = cur_cls
        person_id = 22
        if np.all(det_res == 0):
            return img
        else:
            idx = np.where(det_res[:,0] == id)
            all_id = list(idx[0])
            if len(idx[0])!=0:
                cls_list = list(det_res[idx[0],1])
                if float(class_id) in cls_list:
                    cls_appear = cls_list.index(float(class_id))
                    box = det_res[all_id[cls_appear], 3:]
                    img = img.crop((box[0],box[1],box[2],box[3]))
                else:
                    n = np.argmax(det_res[idx,2], axis=1)
                    box = det_res[idx[0][n[0]], 3:]
                    img = img.crop((box[0],box[1],box[2],box[3]))
            return img
    
    def _load_frame_det_ave(self, path, id, det_res, cur_cls):
        img = Image.open(path).convert('RGB')
        # get box
        class_id = cur_cls
        person_id = 28
        if np.all(det_res == 0):
            return img
        else:
            idx = np.where(det_res[:,0] == id)
            all_id = list(idx[0])
            if len(idx[0])!=0:
                cls_list = list(det_res[idx[0],1])
                if float(class_id) in cls_list:
                    cls_appear = cls_list.index(float(class_id))
                    box = det_res[all_id[cls_appear], 3:]
                    img = img.crop((box[0],box[1],box[2],box[3]))
                else:
                    n = np.argmax(det_res[idx,2], axis=1)
                    box = det_res[idx[0][n[0]], 3:]
                    img = img.crop((box[0],box[1],box[2],box[3]))
            return img
        
    def _load_frame(self, path):
        img = Image.open(path).convert('RGB')
        return img
    def _load_frame_21(self, path):
        img = Image.open(path).convert('RGB')
        return img
    def _load_frame_ave(self, path):
        img = Image.open(path).convert('RGB')
        return img

    def _stft(self, audio):
        spec = librosa.stft(
            audio, n_fft=self.stft_frame, hop_length=self.stft_hop)
        amp = np.abs(spec)
        phase = np.angle(spec)
        return torch.from_numpy(amp), torch.from_numpy(phase)

    def _load_audio_file(self, path):
        if path.endswith('.mp3'):
            audio_raw, rate = torchaudio.load(path)
            audio_raw = audio_raw.numpy().astype(np.float32)

            # range to [-1, 1]
            audio_raw *= (2.0**-31)

            # convert to mono
            if audio_raw.shape[1] == 2:
                audio_raw = (audio_raw[:, 0] + audio_raw[:, 1]) / 2
            else:
                audio_raw = audio_raw[:, 0]
        else:
            audio_raw, rate = librosa.load(path, sr=None, mono=True)

        return audio_raw, rate

    def _load_audio(self, path, center_timestamp, nearest_resample=False):
        audio = np.zeros(self.audLen, dtype=np.float32)

        # silent
        if path.endswith('silent'):
            return audio

        # load audio
        audio_raw, rate = self._load_audio_file(path)

        # repeat if audio is too short
        if audio_raw.shape[0] < rate * self.audSec:
            n = int(rate * self.audSec / audio_raw.shape[0]) + 1
            audio_raw = np.tile(audio_raw, n)

        # resample
        if rate > self.audRate:
            # print('resmaple {}->{}'.format(rate, self.audRate))
            if nearest_resample:
                audio_raw = audio_raw[::rate//self.audRate]
            else:
                audio_raw = librosa.resample(audio_raw, rate, self.audRate)

        # crop N seconds
        len_raw = audio_raw.shape[0]
        center = int(center_timestamp * self.audRate)
        start = max(0, center - self.audLen // 2)
        end = min(len_raw, center + self.audLen // 2)

        audio[self.audLen//2-(center-start): self.audLen//2+(end-center)] = \
            audio_raw[start:end]

        # randomize volume
        if self.split == 'train':
            scale = random.random() + 0.5     # 0.5-1.5
            audio *= scale
        audio[audio > 1.] = 1.
        audio[audio < -1.] = -1.

        return audio

    def _mix_n_and_stft(self, audios):
        N = len(audios)
        mags = [None for n in range(N)]
        audio_mix = np.asarray(audios).sum(axis=0)
        # STFT
        amp_mix, phase_mix = self._stft(audio_mix)
        for n in range(N):
            ampN, _ = self._stft(audios[n])
            mags[n] = ampN.unsqueeze(0)
        # to tensor
        for n in range(N):
            audios[n] = torch.from_numpy(audios[n])
        return amp_mix.unsqueeze(0), mags, phase_mix.unsqueeze(0)

    def dummy_mix_data(self, N):
        frames = [None for n in range(N)]
        audios = [None for n in range(N)]
        mags = [None for n in range(N)]
 
        amp_mix = torch.zeros(1, self.HS, self.WS)
        phase_mix = torch.zeros(1, self.HS, self.WS)

        for n in range(N):
            frames[n] = torch.zeros(
                3, self.num_frames, self.imgSize, self.imgSize)
            audios[n] = torch.zeros(self.audLen)
            mags[n] = torch.zeros(1, self.HS, self.WS)
        return amp_mix, mags, frames, audios, phase_mix
    
    def dummy_mix_data_ave(self, N):
        frames = [None for n in range(N)]
        audios = [None for n in range(N)]
        mags = [None for n in range(N)]
        WS = 256
        amp_mix = torch.zeros(1, self.HS, WS)
        phase_mix = torch.zeros(1, self.HS, WS)

        for n in range(N):
            frames[n] = torch.zeros(
                3, self.num_frames, self.imgSize, self.imgSize)
            audios[n] = torch.zeros(self.audLen)
            mags[n] = torch.zeros(1, self.HS, WS)
        return amp_mix, mags, frames, audios, phase_mix

class BaseDataset(torchdata.Dataset):
    def __init__(self, list_sample, opt, max_sample=-1, split='train'):
        # params
        self.num_frames = opt.num_frames
        self.stride_frames = opt.stride_frames
        self.frameRate = opt.frameRate
        self.imgSize = opt.imgSize
        self.audRate = opt.audRate
        self.audLen = opt.audLen
        self.audSec = 1. * self.audLen / self.audRate
        self.binary_mask = opt.binary_mask

        # STFT params
        self.log_freq = opt.log_freq
        self.stft_frame = opt.stft_frame
        self.stft_hop = opt.stft_hop
        self.HS = opt.stft_frame // 2 + 1
        self.WS = (self.audLen + 1) // self.stft_hop

        self.split = split
        self.seed = opt.seed
        random.seed(self.seed)

        # initialize video transform
        self._init_vtransform()

        # list_sample can be a python list or a csv file of list
        if isinstance(list_sample, str):
            self.list_sample = []
            for row in csv.reader(open(list_sample, 'r'), delimiter=','):
                if len(row) < 2:
                    continue
                self.list_sample.append(row)
        elif isinstance(list_sample, list):
            self.list_sample = list_sample
        else:
            raise('Error list_sample!')

        if self.split == 'train':
            self.list_sample *= opt.dup_trainset
            random.shuffle(self.list_sample)

        if max_sample > 0:
            self.list_sample = self.list_sample[0:max_sample]

        num_sample = len(self.list_sample)
        assert num_sample > 0
        print('# samples: {}'.format(num_sample))

    def __len__(self):
        return len(self.list_sample)

    # video transform funcs
    def _init_vtransform(self):
        transform_list = []
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        if self.split == 'train':
            transform_list.append(vtransforms.Resize(int(self.imgSize * 1.1), Image.BICUBIC))
            transform_list.append(vtransforms.RandomCrop(self.imgSize))
            transform_list.append(vtransforms.RandomHorizontalFlip())
        else:
            transform_list.append(vtransforms.Resize(self.imgSize, Image.BICUBIC))
            transform_list.append(vtransforms.CenterCrop(self.imgSize))

        transform_list.append(vtransforms.ToTensor())
        transform_list.append(vtransforms.Normalize(mean, std))
        transform_list.append(vtransforms.Stack())
        self.vid_transform = transforms.Compose(transform_list)

    # image transform funcs, deprecated
    def _init_transform(self):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        if self.split == 'train':
            self.img_transform = transforms.Compose([
                transforms.Scale(int(self.imgSize)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])
        else:
            self.img_transform = transforms.Compose([
                transforms.Scale(self.imgSize),
                transforms.CenterCrop(self.imgSize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])

    def _load_frames(self, paths):
        frames = []
        for path in paths:
            frames.append(self._load_frame(path))
        frames = self.vid_transform(frames)
        return frames

    def _load_frames_21(self, paths):
        frames = []
        for path in paths:
            frames.append(self._load_frame_21(path))
        frames = self.vid_transform(frames)
        return frames

    def _load_frames_det(self, paths, path_frames_ids,  path_frames_det):
        det_res = np.load(path_frames_det)
        frames = []
        N = len(paths)
        for n in range(N):
            path = paths[n]
            id = path_frames_ids[n]
            frames.append(self._load_frame_det(path, id, det_res))
        frames = self.vid_transform(frames)
        return frames
    def _load_frames_det_21(self, paths, path_frames_ids,  path_frames_det, current_cls):
        det_res = np.load(path_frames_det)
        frames = []
        N = len(paths)
        for n in range(N):
            path = paths[n]
            id = path_frames_ids[n]
            frames.append(self._load_frame_det_21(path, id, det_res, current_cls))
        frames = self.vid_transform(frames)
        return frames
    #pooling alongside the temporal dimension: C, T, H, W -> C, H, W
    def temporal_pooling(self, x):
        C, T, H, W = x.shape
        x = x.permute(0,2,3,1)
        x = x.view(C,H*W,T)
        x = F.adaptive_max_pool1d(x, 1)
        x = x.view(C, H, W)
        return x
    #motion loading

    #C, T, H, W -> C, T spatial pooling
    def spatial_pooling(self, x):
        C, T, H, W = x.shape
        x = F.adaptive_max_pool2d(x, 1)
        x = x.view(C, T)
        return x
    
    def _load_motion_tensor_time(self, path):
        motion = np.load(path)
        x = torch.tensor(motion)
        x = self.spatial_pooling(x)
        return x
    
    def _load_motion_tensor(self, path):
        motion = np.load(path)
        x = torch.tensor(motion)
        x = self.temporal_pooling(x)
        return x

    def _load_frame_det(self, path, id, det_res):

        # load image
        img = Image.open(path).convert('RGB')

        # get box
        idx = np.where(det_res[:, 0] == id)
        if len(idx[0])!=0:
            n = np.argmax(det_res[idx, 2], axis=1)
            bb = det_res[idx[0][n[0]], 3:]
            # crop image
            img = img.crop((bb[0], bb[1], bb[2], bb[3]))
        return img

    def _load_frame_det_21(self, path, id, det_res, cur_cls):

        # load image
        img = Image.open(path).convert('RGB')

        # get box
        class_id = cur_cls + 1
        person_id = 22
        if np.all(det_res == 0):
            return img
        else:
            idx = np.where(det_res[:,0] == id)
            all_id = list(idx[0])
            if len(idx[0])!=0:
                cls_list = list(det_res[idx[0],1])
                if float(class_id) in cls_list:
                    cls_appear = cls_list.index(float(class_id))
                    box = det_res[all_id[cls_appear], 3:]
                    img = img.crop((box[0],box[1],box[2],box[3]))
                else:
                    n = np.argmax(det_res[idx,2], axis=1)
                    box = det_res[idx[0][n[0]], 3:]
                    img = img.crop((box[0],box[1],box[2],box[3]))
            return img

    def _load_frame(self, path):
        img = Image.open(path).convert('RGB')
        return img
    def _load_frame_21(self, path):
        img = Image.open(path).convert('RGB')
        return img

    def _stft(self, audio):
        spec = librosa.stft(
            audio, n_fft=self.stft_frame, hop_length=self.stft_hop)
        amp = np.abs(spec)
        phase = np.angle(spec)
        return torch.from_numpy(amp), torch.from_numpy(phase)

    def _load_audio_file(self, path):
        if path.endswith('.mp3'):
            audio_raw, rate = torchaudio.load(path)
            audio_raw = audio_raw.numpy().astype(np.float32)

            # range to [-1, 1]
            audio_raw *= (2.0**-31)

            # convert to mono
            if audio_raw.shape[1] == 2:
                audio_raw = (audio_raw[:, 0] + audio_raw[:, 1]) / 2
            else:
                audio_raw = audio_raw[:, 0]
        else:
            audio_raw, rate = librosa.load(path, sr=None, mono=True)
            '''
            audio_raw, rate = torchaudio.load(path)
            if audio_raw.size(0)>1:
                audio_raw = torch.mean(audio_raw, dim=0)
            audio_raw = audio_raw.view(-1)
            audio_raw = audio_raw.numpy().astype(np.float32)
            '''

        return audio_raw, rate

    def _load_audio(self, path, center_timestamp, nearest_resample=False):
        audio = np.zeros(self.audLen, dtype=np.float32)

        # silent
        if path.endswith('silent'):
            return audio

        # load audio
        audio_raw, rate = self._load_audio_file(path)

        # repeat if audio is too short
        if audio_raw.shape[0] < rate * self.audSec:
            n = int(rate * self.audSec / audio_raw.shape[0]) + 1
            audio_raw = np.tile(audio_raw, n)

        # resample
        if rate > self.audRate:
            # print('resmaple {}->{}'.format(rate, self.audRate))
            if nearest_resample:
                audio_raw = audio_raw[::rate//self.audRate]
            else:
                audio_raw = librosa.resample(audio_raw, rate, self.audRate)

        # crop N seconds
        len_raw = audio_raw.shape[0]
        center = int(center_timestamp * self.audRate)
        start = max(0, center - self.audLen // 2)
        end = min(len_raw, center + self.audLen // 2)

        audio[self.audLen//2-(center-start): self.audLen//2+(end-center)] = \
            audio_raw[start:end]

        # randomize volume
        if self.split == 'train':
            scale = random.random() + 0.5     # 0.5-1.5
            audio *= scale
        audio[audio > 1.] = 1.
        audio[audio < -1.] = -1.

        return audio

    def _mix_n_and_stft(self, audios):
        N = len(audios)
        mags = [None for n in range(N)]
        for n in range(N):
            audios[n] /= N
        audio_mix = np.asarray(audios).sum(axis=0)

        # STFT
        amp_mix, phase_mix = self._stft(audio_mix)
        for n in range(N):
            ampN, _ = self._stft(audios[n])
            mags[n] = ampN.unsqueeze(0)
        
        # to tensor
        for n in range(N):
            audios[n] = torch.from_numpy(audios[n])
        return amp_mix.unsqueeze(0), mags, phase_mix.unsqueeze(0)

    def dummy_mix_data(self, N):
        frames = [None for n in range(N)]
        audios = [None for n in range(N)]
        mags = [None for n in range(N)]

        amp_mix = torch.zeros(1, self.HS, self.WS)
        phase_mix = torch.zeros(1, self.HS, self.WS)

        for n in range(N):
            frames[n] = torch.zeros(
                3, self.num_frames, self.imgSize, self.imgSize)
            audios[n] = torch.zeros(self.audLen)
            mags[n] = torch.zeros(1, self.HS, self.WS)
        return amp_mix, mags, frames, audios, phase_mix
