import argparse, re
from model import *
from dataset_3d import *
from resnet_2d3d import neq_load_customized
from utils import denorm, AverageMeter, save_checkpoint, write_log

import torch.utils.data
import torch.optim as optim
from tensorboardX import SummaryWriter
from tqdm import tqdm
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--num_workers', default=16, type=int)
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
parser.add_argument('--net', default='resnet18', type=str)
parser.add_argument('--final_dim', default=1024, type=int, help='length of vector output from audio/video subnetwork')
parser.add_argument('--wd', default=1e-5, type=float, help='weight decay')
parser.add_argument('--resume', default='', type=str, help='path of model to resume training')
parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('--reset_lr', action='store_true', help='Reset learning rate when resume training?')
parser.add_argument('--img_dim', default=224, type=int)
parser.add_argument('--out_dir', default='output', type=str, help='Output directory containing Deepfake_data')
parser.add_argument('--print_freq', default=10, type=int, help='frequency of printing output during training')
parser.add_argument('--hyper_param', default=0.99, type=float, help='margin hyper parameter used in loss equation')
parser.add_argument('--threshold', default=0.3, type=float, help='threshold for testing')
parser.add_argument('--test', default='', type=str)
parser.add_argument('--dropout', default=0.5, type=float)
parser.add_argument('--half_network', action='store_true', help='Use shallower network')
parser.add_argument('--audio_format', default='mfcc', choices=['mfcc','waveform'], type=str, help='audio input format')

parser.add_argument('--dataset_size', type=str, default='normal', choices=['normal', 'small'], help='Use small dataset')
parser.add_argument('--dataset', type=str, default='dfdc', choices=['dfdc', 'fakeavceleb'], help='Dataset selection')

parser.add_argument('--mean_separation_loss_weight', default=0, type=float)
parser.add_argument('--vid_ce_loss_weight', default=1, type=float)
parser.add_argument('--aud_ce_loss_weight', default=1, type=float)
parser.add_argument('--distance_loss_weight', default=1, type=float)
parser.add_argument('--kl_loss_weight', default=0, type=float)


def load_network(args):
    if args.audio_format == 'mfcc':
        model = Audio_RNN(img_dim=args.img_dim, network=args.net, num_layers_in_fc_layers=args.final_dim,
                          dropout=args.dropout, full_network=not args.half_network)
    else:
        assert args.audio_format == 'waveform'
        model = Raw_Audio_RNN(img_dim=args.img_dim, network=args.net, num_layers_in_fc_layers=args.final_dim,
                              dropout=args.dropout, full_network=not args.half_network)
    return model   # sum(p.numel() for p in model.parameters())


def load_data(transform, args, mode='train'):
    dataset = deepfake_3d(out_dir=args.out_dir, mode=mode,
                          transform=transform, dataset_size=args.dataset_size,
                          dataset_name=args.dataset)
    return dataset

def get_data(transform, args, mode='train'):
    print('Loading data for "%s" ...' % mode)
    dataset = load_data(transform, args, mode)

    sampler = data.RandomSampler(dataset)

    if mode == 'train':
        data_loader = data.DataLoader(dataset,
                                      batch_size=args.batch_size,
                                      sampler=sampler,
                                      shuffle=False,
                                      num_workers=args.num_workers,
                                      pin_memory=True,
                                      drop_last=True,
                                      collate_fn=my_collate)
    elif mode == 'test':
        data_loader = data.DataLoader(dataset,
                                      batch_size=1,
                                      sampler=sampler,
                                      shuffle=False,
                                      num_workers=args.num_workers,
                                      pin_memory=True,
                                      collate_fn=my_collate)
    else:
        print('No mode ', mode)
        sys.exit()
    print('"%s" dataset size: %d' % (mode, len(dataset)))
    return data_loader


def load_rawaudio_data(transform, args, mode='train'):
    dataset = deepfake_3d_rawaudio(out_dir=args.out_dir, mode=mode,
                                   transform=transform, dataset_size=args.dataset_size,
                                   dataset_name=args.dataset)
    return dataset

def get_rawaudio_data(transform, args, mode='train'):
    print('Loading data for "%s" ...' % mode)
    dataset = load_rawaudio_data(transform, args, mode)

    sampler = data.RandomSampler(dataset)

    if mode == 'train':
        data_loader = data.DataLoader(dataset,
                                      batch_size=args.batch_size,
                                      sampler=sampler,
                                      shuffle=False,
                                      num_workers=args.num_workers,
                                      pin_memory=True,
                                      drop_last=True,
                                      collate_fn=my_collate_rawaudio)
    elif mode == 'test':
        data_loader = data.DataLoader(dataset,
                                      batch_size=1,
                                      sampler=sampler,
                                      shuffle=False,
                                      num_workers=args.num_workers,
                                      pin_memory=True,
                                      collate_fn=my_collate_rawaudio)

    else:
        print('No mode ', mode)
        sys.exit()
    print('"%s" dataset size: %d' % (mode, len(dataset)))
    return data_loader


def set_path(args):
    if args.resume:
        exp_path = os.path.dirname(os.path.dirname(args.resume))
    else:
        exp_path = 'log_tmp/v5_deepfake_audio-'
        if args.dataset != 'dfdc':
            exp_path += args.dataset + '-'
        if args.audio_format == 'waveform':
            exp_path += 'waveform-'
        if args.distance_loss_weight != 1:
            exp_path += 'dl' + str(args.distance_loss_weight) + '-'
        if args.vid_ce_loss_weight != 1:
            exp_path += 'vce' + str(args.vid_ce_loss_weight) + '-'
        if args.aud_ce_loss_weight != 1:
            exp_path += 'ace' + str(args.aud_ce_loss_weight) + '-'
        if args.mean_separation_loss_weight > 0:
            exp_path += 'msl' + str(args.mean_separation_loss_weight)
            exp_path += '-'
        if args.kl_loss_weight > 0:
            exp_path += 'kll' + str(args.kl_loss_weight)
            exp_path += '-'

        if args.dataset_size == 'small':
            exp_path += 'smalldb-'

        exp_path += '{args.img_dim}_{0}_bs{args.batch_size}_lr{1}'.format(
            'r%s' % args.net[6::], \
            args.old_lr if args.old_lr is not None else args.lr, \
            args=args)

    if args.half_network and '_half' not in args.resume:
        exp_path += '_half'

    img_path = os.path.join(exp_path, 'img')
    model_path = os.path.join(exp_path, 'model')
    print('exp_path:', exp_path)
    if not os.path.exists(img_path): os.makedirs(img_path)
    if not os.path.exists(model_path): os.makedirs(model_path)
    return img_path, model_path, os.path.split(exp_path)[-1]

def get_scores(dissimilarity_score_dict, number_of_chunks_dict, target_dict):
    sum_score_fake = 0
    sum_score_real = 0
    total_fake = 0
    total_real = 0
    for video, score in dissimilarity_score_dict.items():
        avg_score = score / number_of_chunks_dict[video]
        if target_dict[video] == 0:
            sum_score_fake += avg_score
            total_fake += 1
        else:
            sum_score_real += avg_score
            total_real += 1
    avg_score_real = sum_score_real / total_real
    avg_score_fake = sum_score_fake / total_fake
    return avg_score_real, avg_score_fake



def main(args):
    torch.manual_seed(0)
    np.random.seed(0)
    global cuda
    cuda = torch.device('cuda')

    if args.dataset != 'dfdc':
        assert args.dataset_size == 'normal'

    model = load_network(args)

    model = model.cuda(0)
    model = nn.DataParallel(model)
    global criterion
    criterion = nn.CrossEntropyLoss()

    print('\n===========Check Grad============')
    for name, param in model.named_parameters():
        print(name, param.requires_grad)
    print('=================================\n')

    params = model.parameters()
    optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.wd)
    least_loss = 0
    global iteration
    iteration = 0

    if args.test:
        if os.path.isfile(args.test):
            print("=> loading testing checkpoint '{}'".format(args.test))
            checkpoint = torch.load(args.test)
            try:
                model.load_state_dict(checkpoint['state_dict'])
            except:
                print('=> [Warning]: weight structure is not equal to test model; Use non-equal load ==')
                sys.exit()
            print("=> loaded testing checkpoint '{}' (epoch {})".format(args.test, checkpoint['epoch']))
            global num_epoch;
            num_epoch = checkpoint['epoch']
        elif args.test == 'random':
            print("=> [Warning] loaded random weights")
        else:
            raise ValueError()

        # test result save folder
        paths = args.test.split('/')
        if args.dataset_size == 'small':
            save_folder = os.path.join('test_results', paths[1], paths[-1][:-8], "small")
        else:
            save_folder = os.path.join('test_results', paths[1], paths[-1][:-8], "normal")
        if not os.path.exists(save_folder): os.makedirs(save_folder)

        transform = transforms.Compose([
            Scale(size=(args.img_dim, args.img_dim)),
            ToTensor(),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        if 'waveform-' in args.test:
            test_loader = get_rawaudio_data(transform, args, 'test')
            train_loader = get_rawaudio_data(transform, args, 'train')
        else:
            test_loader = get_data(transform, args, 'test')
            train_loader = get_data(transform, args, 'train')
        global test_dissimilarity_score
        test_dissimilarity_score = {}
        global test_target
        test_target = {}
        global test_number_ofile_number_of_chunks
        test_number_ofile_number_of_chunks = {}

        test(test_loader, model, args, with_test_data=True, save_folder=save_folder)

        dataset = '' if args.dataset == 'dfdc' else args.dataset
        file_dissimilarity_score = open(os.path.join(save_folder, dataset + "file_dissimilarity_score.pkl"), "wb")
        pickle.dump(test_dissimilarity_score, file_dissimilarity_score)
        file_dissimilarity_score.close()
        file_target = open(os.path.join(save_folder, dataset + "file_target.pkl"), "wb")
        pickle.dump(test_target, file_target)
        file_target.close()
        file_number_of_chunks = open(os.path.join(save_folder, dataset + "file_number_of_chunks.pkl"), "wb")
        pickle.dump(test_number_ofile_number_of_chunks, file_number_of_chunks)
        file_number_of_chunks.close()

        del test_dissimilarity_score
        del test_target
        del test_number_ofile_number_of_chunks

        global train_dissimilarity_score;
        train_dissimilarity_score = {}
        global train_target;
        train_target = {}
        global train_number_ofile_number_of_chunks;
        train_number_ofile_number_of_chunks = {}

        test(train_loader, model, args, with_test_data=False, save_folder=save_folder)

        dataset = '' if args.dataset == 'dfdc' else args.dataset
        file_dissimilarity_score_train = open(os.path.join(save_folder, dataset + "file_dissimilarity_score_train.pkl"), "wb")
        pickle.dump(train_dissimilarity_score, file_dissimilarity_score_train)
        file_dissimilarity_score_train.close()
        file_target_train = open(os.path.join(save_folder, dataset + "file_target_train.pkl"), "wb")
        pickle.dump(train_target, file_target_train)
        file_target_train.close()
        file_number_of_chunks_train = open(os.path.join(save_folder, dataset + "file_number_of_chunks_train.pkl"), "wb")
        pickle.dump(train_number_ofile_number_of_chunks, file_number_of_chunks_train)
        file_number_of_chunks_train.close()

        sys.exit()
    else:  # not test
        torch.backends.cudnn.benchmark = True

    if args.resume:
        if os.path.isfile(args.resume):
            if '_half' in args.resume:
                args.old_lr = float(re.search('_lr(.+?)_half/', args.resume).group(1))
            else:
                args.old_lr = float(re.search('_lr(.+?)/', args.resume).group(1))
            print(args.old_lr)
            print("=> loading resumed checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
            args.start_epoch = checkpoint['epoch']
            iteration = checkpoint['iteration']
            least_loss = checkpoint['least_loss']
            try:
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
            except:
                print('=> [Warning]: weight structure is not equal to checkpoint; Use non-equal load ==')
                model = neq_load_customized(model, checkpoint['state_dict'])
                sys.exit(1)

            print("=> loaded resumed checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("[Warning] no checkpoint found at '{}'".format(args.resume))
            sys.exit(1)

    transform = transforms.Compose([
        Scale(size=(args.img_dim, args.img_dim)),
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # make it -1 to 1 instead of using imagenet statistics
    ])

    if args.audio_format == 'mfcc':
        train_loader = get_data(transform, args, 'train')
    else:
        assert args.audio_format == 'waveform'
        train_loader = get_rawaudio_data(transform, args, 'train')

    global total_sum_scores_real;
    total_sum_scores_real = 0
    global total_sum_scores_fake;
    total_sum_scores_fake = 0
    global count_real;
    count_real = 0
    global count_fake;
    count_fake = 0
    # setup tools
    global de_normalize;
    de_normalize = denorm(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    global img_path;
    img_path, model_path, _ = set_path(args)
    global writer_train
    try:  # old version
        writer_train = SummaryWriter(log_dir=os.path.join(img_path, 'train'))
    except:  # v1.7
        writer_train = SummaryWriter(logdir=os.path.join(img_path, 'train'))

    ### main loop ###
    for epoch in range(args.start_epoch, args.epochs):
        train_loss, avg_score_real, avg_score_fake = train(train_loader, model, optimizer, epoch, args)

        writer_train.add_scalar('global/loss', train_loss, epoch)
        writer_train.add_scalar('global/avg_score_fake', avg_score_fake, epoch)
        writer_train.add_scalar('global/avg_score_real', avg_score_real, epoch)

        # save check_point
        if epoch == 0:
            least_loss = train_loss
        is_best = train_loss <= least_loss
        least_loss = min(least_loss, train_loss)

        save_checkpoint({
            'epoch': epoch + 1,
            'net': args.net,
            'state_dict': model.state_dict(),
            'least_loss': least_loss,
            'avg_score_real': avg_score_real,
            'avg_score_fake': avg_score_fake,
            'optimizer': optimizer.state_dict(),
            'iteration': iteration
        }, is_best, filename=os.path.join(model_path, 'epoch%s.pth.tar' % str(epoch + 1)), keep_all=False)

    print('Training from ep %d to ep %d finished' % (args.start_epoch, args.epochs))


def train(data_loader, model, optimizer, epoch, args):
    losses = AverageMeter()
    ace_losses = AverageMeter()
    vce_losses = AverageMeter()
    dist_losses = AverageMeter()
    meanseparation_losses = AverageMeter()
    kl_losses = AverageMeter()
    model.train()
    global iteration
    dissimilarity_score_dict = {}
    target_dict = {}
    number_of_chunks_dict = {}
    for idx, (video_seq, audio_seq, target, audiopath) in enumerate(data_loader):
        tic = time.time()
        video_seq = video_seq.to(cuda)
        audio_seq = audio_seq.to(cuda)
        target = target.to(cuda)
        B = video_seq.size(0)

        vid_class, aud_class, loss1, vid_out, aud_out, mean_separation_loss, kl_loss = model(video_seq, audio_seq, args.hyper_param, target,
                                                                                       calc_mean_separation_loss=args.mean_separation_loss_weight > 0,
                                                                                       calc_kl_loss=args.kl_loss_weight > 0)

        del video_seq
        del audio_seq

        # acc = calc_accuracy(vid_out, aud_out, target, args.threshold)

        loss = args.distance_loss_weight * loss1  # loss1 seems very very big compared to loss2 and loss3 (at least on the first iteration)
        dist_losses.update(loss.item(), B)
        if args.vid_ce_loss_weight > 0:
            vce_loss = args.vid_ce_loss_weight * criterion(vid_class, target.view(-1))  # cross entropy loss
            loss += vce_loss
            vce_losses.update(vce_loss.item(), B)
        if args.aud_ce_loss_weight > 0:
            ace_loss = args.aud_ce_loss_weight * criterion(aud_class, target.view(-1))  # cross entropy loss
            loss += ace_loss
            ace_losses.update(ace_loss.item(), B)
        if args.mean_separation_loss_weight > 0:
            loss += args.mean_separation_loss_weight * mean_separation_loss
            meanseparation_losses.update(mean_separation_loss.item(), B)
        if args.kl_loss_weight > 0:
            loss += args.kl_loss_weight * kl_loss
            kl_losses.update(kl_loss.item(), B)

        losses.update(loss.item(), B)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pwdist = torch.nn.PairwiseDistance(p=2)
        distance_batch = pwdist(vid_out[0].view(B, -1), aud_out[0].view(B,-1))  # 2-norm distance between video and audio features. [0] means only the last vid_out and aud_out
        total_distance = distance_batch.detach().cpu()

        for batch in range(B):
            vid_name = audiopath[batch].split('/')[-2]
            tar = target[batch, :].view(-1).item()
            dist = total_distance[batch].item()
            if (dissimilarity_score_dict.get(vid_name)):
                dissimilarity_score_dict[vid_name] += dist
                number_of_chunks_dict[vid_name] += 1
            else:
                dissimilarity_score_dict[vid_name] = dist
                number_of_chunks_dict[vid_name] = 1

            if (target_dict.get(vid_name)):
                pass
            else:
                target_dict[vid_name] = tar

        if idx % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time iter: {3}\t'
                  'Loss {loss.val:.4f} ({loss.local_avg:.4f})\t'.format(
                epoch, idx, len(data_loader), time.time() - tic,
                loss=losses))

            total_weight = 0.0
            decay_weight = 0.0
            for m in model.parameters():
                if m.requires_grad: decay_weight += m.norm(2).data.cuda(0)
                total_weight += m.norm(2).data.cuda(0)
            print('Decay weight / Total weight: %.3f/%.3f' % (decay_weight, total_weight))

            writer_train.add_scalar('local/loss', losses.val, iteration)

            if args.aud_ce_loss_weight > 0:
                writer_train.add_scalar('local/ace_loss', ace_losses.val, iteration)

            if args.vid_ce_loss_weight > 0:
                writer_train.add_scalar('local/vce_loss', vce_losses.val, iteration)

            if args.mean_separation_loss_weight > 0:
                writer_train.add_scalar('local/meanseparation_loss', meanseparation_losses.val, iteration)

            if args.kl_loss_weight > 0:
                writer_train.add_scalar('local/kl_loss', kl_losses.val, iteration)

            writer_train.add_scalar('local/dist_loss', dist_losses.val, iteration)

            iteration += 1

    avg_score_real, avg_score_fake = get_scores(dissimilarity_score_dict, number_of_chunks_dict, target_dict)
    return losses.avg, avg_score_real, avg_score_fake


def test(data_loader, model, args, with_test_data=True, save_folder='test_results'):
    losses = AverageMeter()
    model.eval()
    with torch.no_grad():
        dist_fake = []
        dist_real = []
        detail_dist_fake = dict()
        detail_dist_real = dict()

        for idx, (video_seq, audio_seq, target, audiopath) in tqdm(enumerate(data_loader), total=len(data_loader)):
            if len(video_seq) == 0:
                continue

            # just forward the first data in the batch
            video_seq = video_seq[0].unsqueeze(0).to(cuda)
            audio_seq = audio_seq[0].unsqueeze(0).to(cuda)
            target = target[0].unsqueeze(0).to(cuda)
            B = video_seq.size(0)
            assert B == 1

            vid_class, aud_class, loss1, vid_out, aud_out, mean_separation_loss, kl_loss = model(video_seq, audio_seq, args.hyper_param, target,
                                                                                                 calc_mean_separation_loss=args.mean_separation_loss_weight > 0,
                                                                                                 calc_kl_loss=args.kl_loss_weight > 0)

            del video_seq
            del audio_seq

            # LOSS
            loss = loss1
            if args.vid_ce_loss_weight > 0:
                loss += args.vid_ce_loss_weight * criterion(vid_class, target.view(-1))  # cross entropy loss
            if args.aud_ce_loss_weight > 0:
                loss += args.aud_ce_loss_weight * criterion(aud_class, target.view(-1))  # cross entropy loss
            if args.mean_separation_loss_weight > 0:
                loss += args.mean_separation_loss_weight * mean_separation_loss
            losses.update(loss.item(), B)

            tar = target[0, :].view(-1).item()

            pwdist = torch.nn.PairwiseDistance(p=2)
            distance_batch = pwdist(vid_out[0][0].view(1, -1), aud_out[0][0].view(1, -1))  # 2-norm distance between video and audio features. [0] means only the last vid_out and aud_out

            dist = distance_batch[0].detach().cpu()
            if tar == 0:
                dist_fake.append(dist.item())
            else:
                dist_real.append(dist.item())

            vid_name = audiopath[0].split('/')[-2]
            # print(vid_name)
            if with_test_data:
                if (test_dissimilarity_score.get(vid_name)):
                    test_dissimilarity_score[vid_name] += dist
                    test_number_ofile_number_of_chunks[vid_name] += 1
                else:
                    test_dissimilarity_score[vid_name] = dist
                    test_number_ofile_number_of_chunks[vid_name] = 1

                if (test_target.get(vid_name)):
                    pass
                else:
                    test_target[vid_name] = tar

            else:
                if (train_dissimilarity_score.get(vid_name)):
                    train_dissimilarity_score[vid_name] += dist
                    train_number_ofile_number_of_chunks[vid_name] += 1
                else:
                    train_dissimilarity_score[vid_name] = dist
                    train_number_ofile_number_of_chunks[vid_name] = 1

                if (train_target.get(vid_name)):
                    pass
                else:
                    train_target[vid_name] = tar

    print('Loss {loss.avg:.4f}\t'.format(loss=losses))
    write_log(content='Loss {loss.avg:.4f}\t'.format(loss=losses, args=args),
              epoch=num_epoch,
              filename=os.path.join(os.path.dirname(args.test), 'test_log.md'))
    return losses.avg


if __name__ == '__main__':
    args = parser.parse_args()
    args.old_lr = None

    main(args)
