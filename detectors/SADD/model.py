import torch
import torch.nn as nn
import torch.nn.functional as FU
from select_backbone import select_resnet
import math
import timeit



class CNN_Aud(nn.Module):
    def __init__(self, full_network=True):
        super(CNN_Aud, self).__init__()
        self.full_network = full_network

        self.netcnnaud_layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        if full_network:
            self.netcnnaud_layer2 = nn.Sequential(
                nn.MaxPool2d(kernel_size=(1, 1), stride=(1, 1)),
                # this one actually not doing anything, but anyway, it's in the MDS original code so whatever

                nn.Conv2d(64, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.BatchNorm2d(192),
                nn.ReLU(inplace=True),
            )
            self.netcnnaud_layer3 = nn.Sequential(
                nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 2)),

                nn.Conv2d(192, 384, kernel_size=(3, 3), padding=(1, 1)),
                nn.BatchNorm2d(384),
                nn.ReLU(inplace=True),

                nn.Conv2d(384, 256, kernel_size=(3, 3), padding=(1, 1)),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),

                nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1)),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            )
            self.netcnnaud_layer4 = nn.Sequential(
                nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),

                nn.Conv2d(256, 512, kernel_size=(5, 4), padding=(0, 0)),
                nn.BatchNorm2d(512),
            )
        else:
            self.netcnnaud_layer2 = nn.Sequential(
                nn.MaxPool2d(kernel_size=(1, 1), stride=(1, 1)),
                # this one actually not doing anything, but anyway, it's in the MDS original code so whatever

                nn.Conv2d(64, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.BatchNorm2d(192),
            )
            self.netcnnaud_layer3 = nn.Sequential(
                nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 2)),
            )
            self.netcnnaud_layer4 = nn.Sequential(
                nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),
                nn.MaxPool2d(kernel_size=(5, 4), stride=(1, 1)),
            )

    def forward(self, x, return_intermediate=False):
        x1 = self.netcnnaud_layer1(x)
        x2 = self.netcnnaud_layer2(x1)
        x3 = self.netcnnaud_layer3(x2)
        x4 = self.netcnnaud_layer4(x3)
        if return_intermediate:
            return x4, x3, x2, x1
        else:
            return x4


class CNN_RawAud(CNN_Aud):
    def __init__(self, *args, **kwargs):
        super(CNN_RawAud, self).__init__(*args, **kwargs)

        self.netcnnaud_layer1 = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=80, stride=8),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )  # output: batch x 128 x 5991

        if self.full_network:
            self.netcnnaud_layer2 = nn.Sequential(
                nn.MaxPool1d(kernel_size=4, stride=4),

                nn.Conv1d(128, 192, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(192),
                nn.ReLU(inplace=True),
                nn.Conv1d(192, 192, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(192),
                nn.ReLU(inplace=True),
            )   # output: batch x 192 x 1497
            self.netcnnaud_layer3 = nn.Sequential(
                nn.MaxPool1d(kernel_size=4, stride=4),

                nn.Conv1d(192, 256, kernel_size=3, padding=1),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),

                nn.Conv1d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
            )   # output: batch x 256 x 374
            self.netcnnaud_layer4 = nn.Sequential(
                nn.MaxPool1d(kernel_size=4, stride=4),

                nn.Conv1d(256, 512, kernel_size=4, padding=0),
                nn.BatchNorm1d(512),
            )    # output: batch x 512 x 90
        else:
            self.netcnnaud_layer2 = nn.Sequential(
                nn.MaxPool1d(kernel_size=4, stride=4),

                nn.Conv1d(128, 192, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(192),
                nn.ReLU(inplace=True),
                nn.Conv1d(192, 192, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(192),
            )  # output: batch x 192 x 1497
            self.netcnnaud_layer3 = nn.Sequential(
                nn.MaxPool1d(kernel_size=4, stride=4),
            )  # output: batch x 192 x 374
            self.netcnnaud_layer4 = nn.Sequential(
                nn.MaxPool1d(kernel_size=4, stride=4),
                nn.MaxPool1d(kernel_size=4, stride=1, padding=0),
            )  # output: batch x 192 x 90



class Audio_RNN(nn.Module):
    def __init__(self, img_dim, network='resnet50', num_layers_in_fc_layers=1024, dropout=0.5, full_network=True):
        super(Audio_RNN, self).__init__();

        self.__nFeatures__ = 24;
        self.__nChs__ = 32;
        self.__midChs__ = 32;
        self.aud2visspace = None

        self.full_network = full_network

        self.netcnnaud = CNN_Aud(full_network=full_network)
        self.netcnnaud_finalrelu = nn.ReLU()

        if full_network:
            last_aud_feat = 512
        else:
            last_aud_feat = 192

        self.netfcaud = nn.Sequential(
            nn.Linear(last_aud_feat * 21, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Linear(4096, num_layers_in_fc_layers),
        )

        self.netcnnlip, self.param = select_resnet(network, track_running_stats=False, full_resnet=full_network)
        self.last_size = int(math.ceil(img_dim / 32))
        if full_network:
            self.last_duration = int(math.ceil(30 / 4))
        else:
            self.last_duration = int(math.ceil(30 / 2))

        self.netfclip = nn.Sequential(
            nn.Linear(self.param['feature_size'] * self.last_size * self.last_size, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Linear(4096, num_layers_in_fc_layers),
        );

        self.final_bn_lip = nn.BatchNorm1d(num_layers_in_fc_layers)
        self.final_bn_lip.weight.data.fill_(1)
        self.final_bn_lip.bias.data.zero_()

        self.final_fc_lip = nn.Sequential(nn.Dropout(dropout), nn.Linear(num_layers_in_fc_layers, 2))
        self._initialize_weights(self.final_fc_lip)

        self.final_bn_aud = nn.BatchNorm1d(num_layers_in_fc_layers)
        self.final_bn_aud.weight.data.fill_(1)
        self.final_bn_aud.bias.data.zero_()

        self.final_fc_aud = nn.Sequential(nn.Dropout(dropout), nn.Linear(num_layers_in_fc_layers, 2))
        self._initialize_weights(self.final_fc_aud)

        self._initialize_weights(self.netcnnaud.netcnnaud_layer1)
        self._initialize_weights(self.netcnnaud.netcnnaud_layer2)
        self._initialize_weights(self.netcnnaud.netcnnaud_layer3)
        self._initialize_weights(self.netcnnaud.netcnnaud_layer4)
        self._initialize_weights(self.netfcaud)
        self._initialize_weights(self.netfclip)
        if self.aud2visspace is not None:
            for i in range(len(self.aud2visspace)):
                for j in range(len(self.aud2visspace[0])):
                    self._initialize_weights(self.aud2visspace[i][j])

        self.kldivloss_criterion = nn.KLDivLoss(reduction="none", log_target=True)

    def forward(self, video_seq, audio_seq, hyper_param, target,
                calc_mean_separation_loss=False, hyper_param_mean_separation_loss=None, mean_separation_loss_std_detach=False,
                calc_kl_loss=False):

        vid_out = self.forward_lip(video_seq, return_intermediate=True)  # batch x 1024

        aud_out = self.forward_aud(audio_seq, return_intermediate=True)  # batch x 1024

        vid_cls_input = vid_out[0]
        aud_cls_input = aud_out[0]

        vid_class = self.final_classification_lip(vid_cls_input)  # batch x 2

        aud_class = self.final_classification_aud(aud_cls_input)

        loss1, mean_separation_loss, kl_loss = self.calc_loss(vid_out, aud_out, target, hyper_param,
                                                              calc_mean_separation_loss=calc_mean_separation_loss,
                                                              calc_kl_loss=calc_kl_loss)

        return vid_class, aud_class, loss1, vid_out, aud_out, mean_separation_loss, kl_loss

    def _margin_triplet_loss(self, target_batch, distance_batch, hyper_param, power=2):  # Note: there is ** 2
        return ((target_batch * (distance_batch ** power)) + ((1 - target_batch) * (
                torch.max(hyper_param - distance_batch, torch.zeros_like(
                    distance_batch)) ** power)))  # minimize distance if class 1 (real), maximize distance with margin hyper_param if class 0 (fake)


    def calc_loss(self, vid_out, aud_out, target, hyper_param,
                  calc_mean_separation_loss=False,
                  calc_kl_loss=False):
        assert len(vid_out) == len(aud_out)
        batch_size = target.size(0)
        loss = 0
        mean_separation_loss = 0.0
        kl_loss = 0.0

        pwdist = torch.nn.PairwiseDistance(p=2)
        distance_batch = pwdist(vid_out[0].view(batch_size, -1), aud_out[0].view(batch_size, -1))  # 2-norm distance between video and audio features. [0] means only the last vid_out and aud_out (which is the final feature)
        loss_batch = self._margin_triplet_loss(target.view(batch_size), distance_batch, hyper_param)
        loss += torch.sum(loss_batch)
        if calc_mean_separation_loss:
            mean_separation = torch.pow((torch.mean(vid_out[0].view(batch_size, -1), 1) - torch.mean(aud_out[0].view(batch_size, -1), 1)), 2)

            separation_hyperparam = torch.std(vid_out[0]) + torch.std(aud_out[0])
            separation_hyperparam = separation_hyperparam.detach()

            mean_separation_loss += torch.mean(self._margin_triplet_loss(target.view(batch_size), mean_separation, separation_hyperparam, power=1))
        if calc_kl_loss:
            kl_div = self.kldivloss_criterion(FU.log_softmax(vid_out[0].view(batch_size, -1), dim=1), FU.log_softmax(aud_out[0].view(batch_size, -1), dim=1))
            kl_div = torch.sum(kl_div, dim=1)  # KL divergence is sum.
            kl_loss = torch.mean((target.view(batch_size) * kl_div) + \
                      ((1 - target.view(batch_size)) * (-kl_div)))  # minimize distance if class 1 (real), maximize distance with margin hyper_param if class 0 (fake)


        return loss.mul_(1 / batch_size), mean_separation_loss, kl_loss  # averaged based on the batch size

    def forward_aud(self, x, return_intermediate=False):
        (B, N, N, H, W) = x.shape  # batch, 1, 1, 13, 99
        x = x.view(B * N, N, H, W)  # batch x 1 x 13 x 99

        mid_4, mid_3, mid_2, mid_1 = self.netcnnaud(x, return_intermediate=True)  # batch x 512 (192 for half) x 1 x 21, batch x 256 (192 for half) x 11 x 49, batch x 193 x 13 x 99, batch x 64 x 13 x 99
        mid = self.netcnnaud_finalrelu(mid_4)  # batch x 512 x 1 x 21

        mid = mid.view((mid.size()[0], -1))  # batch x 10752 (for full size), batch x 4032 (for half size)
        out = self.netfcaud(mid)  # batch x 1024
        if return_intermediate:
            if self.full_network:
                return out, mid_4, mid_3, mid_2, mid_1
            else:
                return out, mid_2, mid_1
        else:
            return out

    def forward_lip(self, x, return_intermediate=False):
        (B, N, C, NF, H, W) = x.shape
        x = x.view(B * N, C, NF, H, W)  # batch x 3 x 30 x 224 x 224
        if self.full_network:
            feature4, feature3, feature2, feature1 = self.netcnnlip(x,
                                                                    return_intermediate=True)  # batch x 256 x 8 x 7 x 7, , batch x 256 x 15 x 14 x 14, batch x 128 x 30 x 28 x 28, batch x 64 x 30 x 56 x 56
            feature = FU.avg_pool3d(feature4, (self.last_duration, 1, 1),
                                   stride=(1, 1, 1))  # batch x 256 x 1 x 7 x 7 (global avg pool across time domain)
        else:
            feature2, feature1 = self.netcnnlip(x, return_intermediate=True)  # batch x 128 x 15 x 28 x 28, batch x 64 x 30 x 56 x 56
            feature = FU.avg_pool3d(feature2, (self.last_duration, 1, 1),
                                   stride=(1, 1, 1))  # batch x 128 x 1 x 28 x 28 (global avg pool across time domain)
            feature = FU.max_pool3d(feature, (1, 2, 2), stride=(2, 2, 2))  # batch x 128 x 1 x 14 x 14
            feature = FU.max_pool3d(feature, (1, 2, 2), stride=(2, 2, 2))  # batch x 128 x 1 x 7 x 7

        feature = feature.view(B, N, self.param['feature_size'], self.last_size,
                               self.last_size)  # batch x 1 x 256 x 7 x 7 (for full size), batch x 1 x 128 x 7 x 7 (for full size),
        feature = feature.view((feature.size()[0], -1))  # batch x 12544 (for full size), batch x 6272 (for half size)
        out = self.netfclip(feature)  # batch x 1024
        if return_intermediate:
            if self.full_network:
                return out, feature4, feature3, feature2, feature1
            else:
                return out, feature2, feature1
        else:
            return out

    def final_classification_lip(self, feature):
        feature = self.final_bn_lip(feature)
        output = self.final_fc_lip(feature)
        return output

    def final_classification_aud(self, feature):
        feature = self.final_bn_aud(feature)
        output = self.final_fc_aud(feature)
        return output

    def forward_lipfeat(self, x):
        mid = self.netcnnlip(x);
        out = mid.view((mid.size()[0], -1));
        return out;

    def _initialize_weights(self, module):
        for m in module:
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.ReLU) or isinstance(m, nn.MaxPool2d) or isinstance(m, nn.Dropout) or isinstance(m, nn.MaxPool1d) or isinstance(m, nn.Flatten):
                pass
            else:
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None: m.bias.data.zero_()



class Raw_Audio_RNN(Audio_RNN):
    def __init__(self, *args, **kwargs):
        super(Raw_Audio_RNN, self).__init__(*args, **kwargs);

        self.netcnnaud = CNN_RawAud(full_network=kwargs['full_network'])
        self.netcnnaud_finalrelu = nn.ReLU()

        if kwargs['full_network']:
            last_aud_feat = 512
        else:
            last_aud_feat = 192

        self.netfcaud = nn.Sequential(
            nn.Linear(last_aud_feat * 90, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Linear(4096, kwargs['num_layers_in_fc_layers']),
        )

        self.final_bn_aud = nn.BatchNorm1d(kwargs['num_layers_in_fc_layers'])
        self.final_bn_aud.weight.data.fill_(1)
        self.final_bn_aud.bias.data.zero_()

        self.final_fc_aud = nn.Sequential(nn.Dropout(kwargs['dropout']), nn.Linear(kwargs['num_layers_in_fc_layers'], 2))
        self._initialize_weights(self.final_fc_aud)

        self._initialize_weights(self.netcnnaud.netcnnaud_layer1)
        self._initialize_weights(self.netcnnaud.netcnnaud_layer2)
        self._initialize_weights(self.netcnnaud.netcnnaud_layer3)
        self._initialize_weights(self.netcnnaud.netcnnaud_layer4)
        self._initialize_weights(self.netfcaud)
        self._initialize_weights(self.netfclip)

    def forward_aud(self, x, return_intermediate=False):
        (B, C) = x.shape  # batch, 48000
        x = x.view(B, 1, C)  # batch x 1 x 48000

        mid_4, mid_3, mid_2, mid_1 = self.netcnnaud(x, return_intermediate=True)  # batch x 512 (192 for half) x 184, batch x 256 (192 for half) x 748, batch x 192 x 2995, batch x 128 x 11981
        mid = self.netcnnaud_finalrelu(mid_4)

        mid = mid.view((mid.size()[0], -1))  # batch x 94208 (for full size), batch x 35328 (for half size)
        out = self.netfcaud(mid)  # batch x 1024
        if return_intermediate:
            if self.full_network:
                return out, mid_4, mid_3, mid_2, mid_1
            else:
                return out, mid_2, mid_1
        else:
            return out


