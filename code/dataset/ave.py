import os
import random
import numpy as np
import csv
from .base import BaseAVEDataset
class AVEDataset(BaseAVEDataset):
    def __init__(self, list_sample, cur_num_mix, opt, **kwargs):
        super(AVEDataset, self).__init__(
            list_sample, opt, **kwargs)
        self.fps = opt.frameRate
        self.num_mix = cur_num_mix
        self.audLen = opt.audLen

    def __getitem__(self, index):
        N = self.num_mix
        frames = [None for n in range(N)]
        audios = [None for n in range(N)]
        infos = [[] for n in range(N)]
        path_frames = [[] for n in range(N)]
        path_frames_ids = [[] for n in range(N)]
        path_frames_det = ['' for n in range(N)]
        path_audios = ['' for n in range(N)]
        center_frames = [0 for n in range(N)]
        class_list = []


        if self.split == 'train':
            infos[0] = self.list_sample[index]
            cls = infos[0][0].split('/')[1]
            class_list.append(cls)

            for n in range(1, N):
                indexN = random.randint(0, len(self.list_sample)-1)
                sample = self.list_sample[indexN]
                while sample[0].split('/')[1] in class_list:
                    indexN = random.randint(0, len(self.list_sample) - 1)
                    sample = self.list_sample[indexN]
                infos[n] = sample
                class_list.append(sample[0].split('/')[1])
        elif self.split == 'val':
            infos[0] = self.list_sample[index]
            cls = infos[0][0].split('/')[1]
            class_list.append(cls)
            if not self.split == 'train':
                random.seed(index)

            for n in range(1, N):
                indexN = random.randint(0, len(self.list_sample) - 1)
                sample = self.list_sample[indexN]
                while sample[0].split('/')[1] in class_list:
                    indexN = random.randint(0, len(self.list_sample) - 1)
                    sample = self.list_sample[indexN]
                infos[n] = sample
                class_list.append(sample[0].split('/')[1])
        else:
            csv_lis_path = "YOURPATH/data/AVE/testave.csv"
            csv_lis = []
            for row in csv.reader(open(csv_lis_path, 'r'), delimiter=','):
                if len(row) < 2:
                    continue
                csv_lis.append(row)
            random.seed(index) # fixed
            samples = self.list_sample[index]

            for n in range(N):
                sample = samples[n].replace(" ", "")
                for i in range(len(csv_lis)):
                    data = csv_lis[i]
                    if sample in data:
                        infos[n] = data
                        break

            cls = infos[0][0].split('/')[1]
            class_list.append(cls)
            for n in range(1,N):
                class_list.append(infos[n][0].split('/')[1])
        #use for transformer code 
        instrument_dict = {'Accordion':0, 'Acoustic_guitar':1, 'Aircraft':2, 'Baby':3, 'Banjo':4, 'Bark':5, 'Bus':6, 'Cat':7, 'Chainsaw':8, 
        'Church_bell':9, 'Clock':10, 'Female':11, 'Flute':12, 'Food':13, 'Goat':14, 'Helicopter':15, 'Horse':16, 'Male':17, 'Mandolin':18, 
        'Motorcycle':19, 'Race_car':20, 'Rodent':21, 'Shofar':22, 'Toilet':23, 'Train':24, 'Truck':25, 'Ukulele':26, 'Violin':27}
        for i in range(len(class_list)):
            class_list[i] = instrument_dict[class_list[i]]

        # select frames
        idx_margin = max(
            int(self.fps * 1), (self.num_frames // 2) * self.stride_frames)
        for n, infoN in enumerate(infos):
            path_audioN, path_frameN, count_framesN = infoN

            if self.split == 'train':
                # random, not to sample start and end n-frames
                center_frameN = random.randint(
                    idx_margin+1, int(count_framesN)-idx_margin)

            else:
                center_frameN = int(count_framesN) // 2 + 1
            center_frames[n] = center_frameN

            # absolute frame/audio paths
            for i in range(self.num_frames):
                idx_offset = (i - self.num_frames // 2) * self.stride_frames
                path_frames[n].append(
                    os.path.join("YOURPATH/AVE_Dataset/frames",
                        path_frameN[1:],
                        '{:06d}.jpg'.format(center_frameN + idx_offset)))
                path_frames_ids[n].append(center_frameN + idx_offset)
            path_frames_det[n] = os.path.join("YOURPATH/AVE_Dataset/detection_results",
                        path_frameN[1:]+'.npy')

            path_audios[n] = os.path.join("YOURPATH/AVE_Dataset/audio", path_audioN[1:])

        try:
            for n, infoN in enumerate(infos):
                frames[n] = self._load_frames_det_ave(path_frames[n], path_frames_ids[n], path_frames_det[n], class_list[n])    
                # jitter audio
                center_timeN = (center_frames[n] - 0.5) / self.fps
                audios[n] = self._load_audio(path_audios[n], center_timeN)
            mag_mix, mags, phase_mix = self._mix_n_and_stft(audios)

        except Exception as e:
            print('Failed loading frame/audio: {}'.format(e))
            # create dummy data
            mag_mix, mags, frames, audios, phase_mix = \
                self.dummy_mix_data_ave(N)
        
        ret_dict = {'mag_mix': mag_mix, 'frames': frames, 'mags': mags, 'classes': class_list}
        if self.split != 'train':
            ret_dict['audios'] = audios
            ret_dict['phase_mix'] = phase_mix
            ret_dict['infos'] = infos

        return ret_dict