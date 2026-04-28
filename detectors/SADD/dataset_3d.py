import torch
from torch.utils import data
import os
import pandas as pd
from augmentation import *
from scipy.io import wavfile
import python_speech_features
from utils import min_max_normalize

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def my_collate_rawaudio(batch):
    batch = list(filter(lambda x: x is not None and x[1].size()[0] == 48000, batch))
    if len(batch) == 0:
        return [[],[],[],[],[],[],[]]
    return torch.utils.data.dataloader.default_collate(batch)


def my_collate(batch):
    batch = list(filter(lambda x: x is not None and x[1].size()[3] == 99, batch))
    if len(batch) == 0:
        return [[],[],[],[],[],[]]
    return torch.utils.data.dataloader.default_collate(batch)

class deepfake_3d(data.Dataset):
    def __init__(self, out_dir,
                 mode='train',
                 transform=None,
                 dataset_size='normal',
                 dataset_name='dfdc'):
        assert dataset_size in ['normal', 'small']
        assert dataset_name in ['dfdc', 'fakeavceleb']
        self.mode = mode
        self.transform = transform
        self.out_dir = out_dir
        self.dataset_name = dataset_name

        # splits
        if mode == 'train':
            assert dataset_name == 'dfdc'
            if dataset_size == 'normal':
                split = os.path.join(self.out_dir, 'train_split.csv')
            else:
                split = os.path.join(self.out_dir, 'train_split_small.csv')
            video_info = pd.read_csv(split, header=None)
        elif (mode == 'test'):
            if dataset_name == 'dfdc':
                if dataset_size == 'normal':
                    split = os.path.join(self.out_dir, 'test_split.csv')
                else:
                    split = os.path.join(self.out_dir, 'test_split_small.csv')
            else: # elif dataset_name == 'fakeavceleb':
                assert dataset_size == 'normal'
                split = os.path.join(self.out_dir, 'test_fakeavceleb_split.csv')

            video_info = pd.read_csv(split, header=None)
        else:
            raise ValueError('wrong mode')

        # get label list
        self.label_dict_encode = {}
        self.label_dict_decode = {}
        self.label_dict_encode['fake'] = 0
        self.label_dict_decode['0'] = 'fake'
        self.label_dict_encode['real'] = 1
        self.label_dict_decode['1'] = 'real'

        self.video_info = video_info

    def get_vpath_audiopath_label(self, index):
        vpath, audiopath, label = self.video_info.iloc[index]

        return vpath, audiopath, label

    def __getitem__(self, index):
        try:
            vpath, audiopath, label = self.get_vpath_audiopath_label(index)
            vpath = os.path.join(self.out_dir, vpath)
            audiopath = os.path.join(self.out_dir, audiopath)
            seq = [pil_loader(os.path.join(vpath, img)) for img in
                   sorted(os.listdir(vpath))]

            t_seq = self.transform(seq)  # apply same transform

            (C, H, W) = t_seq[0].size()
            t_seq = torch.stack(t_seq, 0)

            sample_rate, audio = wavfile.read(audiopath)  # audio size: 48000 (range, so far checking some samples, can be -ten thousands to + ten thousands)

            if self.dataset_name == 'dfdc':
                audio_label = label
                visual_label = label
            else:
                audio_label = 'real' if 'RealAudio' in vpath else 'fake'
                visual_label = 'real' if 'RealVideo' in vpath else 'fake'

            t_seq = t_seq.view(1, 30, C, H, W).transpose(1, 2)

            mfcc = zip(*python_speech_features.mfcc(audio, sample_rate, nfft=2048))
            mfcc = np.stack([np.array(i) for i in mfcc])

            cc = np.expand_dims(np.expand_dims(mfcc, axis=0), axis=0)
            cct = torch.autograd.Variable(torch.from_numpy(cc.astype(float)).float())

            vid = self.encode_label(label)  # fake = 0; real = 1
            aud_label = self.encode_label(audio_label)
            vis_label = self.encode_label(visual_label)

        except:
            return None

        return t_seq, cct, torch.LongTensor([vid]), audiopath

    def __len__(self):
        return len(self.video_info)

    def encode_label(self, label_name):
        return self.label_dict_encode[label_name]

    def decode_label(self, label_code):
        return self.label_dict_decode[label_code]


class deepfake_3d_rawaudio(deepfake_3d):
    def __getitem__(self, index):
        success = 0
        while success == 0:
            try:
                vpath, audiopath, label = self.get_vpath_audiopath_label(index)
                vpath = os.path.join(self.out_dir, vpath)
                audiopath = os.path.join(self.out_dir, audiopath)

                seq = [pil_loader(os.path.join(vpath, img)) for img in
                       sorted(os.listdir(vpath))]

                sample_rate, audio = wavfile.read(audiopath)

                t_seq = self.transform(seq)  # apply same transform

                (C, H, W) = t_seq[0].size()
                t_seq = torch.stack(t_seq, 0)

                # normalize audio. Bcos seems like each audio data has mean roughly 0, just the range is different (maybe some audio is louder than the others), better to normalize to -1 to 1 based on each data range.
                normalized_raw_audio = min_max_normalize(audio, int(audio.min()), int(audio.max()))
                normalized_raw_audio = torch.autograd.Variable(torch.from_numpy(normalized_raw_audio.astype(float)).float())
                normalized_raw_audio = (normalized_raw_audio - 0.5) / 0.5

                success = 1
            except:
                index = random.randint(0, self.__len__())  # select other data
                print('WARNING: index ', index, ' has something wrong with data fetching')
                continue

        if self.dataset_name == 'dfdc':
            audio_label = label
            visual_label = label
        else:
            audio_label = 'real' if 'RealAudio' in vpath else 'fake'
            visual_label = 'real' if 'RealVideo' in vpath else 'fake'

        t_seq = t_seq.view(1, 30, C, H, W).transpose(1, 2)

        vid = self.encode_label(label)  # fake = 0; real = 1
        aud_label = self.encode_label(audio_label)
        vis_label = self.encode_label(visual_label)

        return t_seq, normalized_raw_audio, torch.LongTensor([vid]), audiopath

