import argparse
import pickle, numpy as np
from sklearn.metrics import roc_auc_score
import os
from utils import min_max_normalize
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--normalize', default='none', type=str, choices=['none', 'train'],
                    help='Min-max normalize the score with statistics of train data. None means just thresholding to 0 or 1')
parser.add_argument('--folder', type=str,
                    help='Folder location of the test results')
parser.add_argument('--dataset', type=str, default='dfdc', choices=['dfdc', 'fakeavceleb'], help='Dataset selection')
args = parser.parse_args()

orig_stdout = sys.stdout
f = open(os.path.join(args.folder, 'output.txt'), 'a')
sys.stdout = f

print('')
print('')
print(args)

print('test on dataset: ', args.dataset)

dataset = '' if args.dataset == 'dfdc' else args.dataset

# file_dissimilarity_score.pkl contains a dictionary with key as video name and value as the sum of
# dissimilarity scores of all chunks of that video 
with open(os.path.join(args.folder, dataset + 'file_dissimilarity_score.pkl'), 'rb') as handle:
    test_dissimilarity_score = pickle.load(handle)

# file_target.pkl contains a dictionary with key as video name and value as the true target (real/fake)
# for that video
with open(os.path.join(args.folder, dataset + 'file_target.pkl'), 'rb') as handle:
    test_target = pickle.load(handle)

# file_number_of_chunks.pkl contains a dictionary with key as video name and value as the number of chunks 
# in that video
with open(os.path.join(args.folder, dataset + 'file_number_of_chunks.pkl'), 'rb') as handle:
    test_number_of_chunks = pickle.load(handle)

# file_dissimilarity_score.pkl contains a dictionary with key as video name and value as the sum of
# dissimilarity scores of all chunks of that video
with open(os.path.join(args.folder, dataset + 'file_dissimilarity_score_train.pkl'), 'rb') as handle:
    train_dissimilarity_score = pickle.load(handle)

# file_target.pkl contains a dictionary with key as video name and value as the true target (real/fake)
# for that video
with open(os.path.join(args.folder, dataset + 'file_target_train.pkl'), 'rb') as handle:
    train_target = pickle.load(handle)

# file_number_of_chunks.pkl contains a dictionary with key as video name and value as the number of chunks
# in that video
with open(os.path.join(args.folder, dataset + 'file_number_of_chunks_train.pkl'), 'rb') as handle:
    train_number_of_chunks = pickle.load(handle)

# calculating statistics of mean dissimilarity score based on training data
score_list = []
for video, score in train_dissimilarity_score.items():
    tar = train_target[video]
    score = train_dissimilarity_score[video]
    score_list.append(score.cpu().item())
    num_chunks = train_number_of_chunks[video]
    mean_dissimilarity_score = (score.item()) / num_chunks
min_mean_dissimilarity_score = np.array(score_list).min()
max_mean_dissimilarity_score = np.array(score_list).max()

thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

print('==============================')

dissimilarity_score = test_dissimilarity_score
target = test_target
number_of_chunks = test_number_of_chunks

for threshold in thresholds:
    y_tar = np.zeros((len(target), 1))
    y_pred = np.zeros((len(target), 1))

    count = 0

    dist_fake = []
    dist_real = []
    pred_fake = []
    pred_real = []

    for video, score in dissimilarity_score.items():
        tar = target[video]
        score = dissimilarity_score[video]
        num_chunks = number_of_chunks[video]
        mean_dissimilarity_score = (score.item()) / num_chunks

        if args.normalize != 'none':
            mean_dissimilarity_score = min_max_normalize(mean_dissimilarity_score, min_mean_dissimilarity_score,
                                                         max_mean_dissimilarity_score)  # high score = fake
            pred = min(1, max(0, 1-mean_dissimilarity_score))

            if tar == 0:
                pred_fake.append(pred)
                dist_fake.append(mean_dissimilarity_score)
            else:
                pred_real.append(pred)
                dist_real.append(mean_dissimilarity_score)

            if mean_dissimilarity_score >= threshold:
                # predicted target is fake
                binary_pred = 0
            else:
                # predicted target is real
                binary_pred = 1


        else:  # elif args.normalize == 'none'
            if tar == 0:
                dist_fake.append(mean_dissimilarity_score)
            else:
                dist_real.append(mean_dissimilarity_score)

            if mean_dissimilarity_score >= threshold:
                # predicted target is fake
                pred = 0
            else:
                # predicted target is real
                pred = 1


        y_tar[count, 0] = tar
        y_pred[count, 0] = pred
        count += 1

    print(args.dataset + '-Video wise AUC with threshold ' + str(threshold) + ' is: ' + str(roc_auc_score(y_tar, y_pred)))

print('==============================')


sys.stdout = orig_stdout
f.close()