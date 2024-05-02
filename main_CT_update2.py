from __future__ import print_function
import sys
import argparse

import numpy as np
import torch
import torch.optim as optim
import torch.utils.data as data_utils
import random
from torch.autograd import Variable
import torch.nn.functional as F
from dataloader import MnistBags
from model_CT_update1 import Attention, GatedAttention
from CT_dataloader import custom_collate,LungDataset

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST bags Example')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                    help='learning rate (default: 0.0005)')
parser.add_argument('--reg', type=float, default=10e-5, metavar='R',
                    help='weight decay')
parser.add_argument('--target_number', type=int, default=9, metavar='T',
                    help='bags have a positive labels if they contain at least one 9')
parser.add_argument('--mean_bag_length', type=int, default=10, metavar='ML',
                    help='average bag length')
parser.add_argument('--var_bag_length', type=int, default=2, metavar='VL',
                    help='variance of bag length')
parser.add_argument('--num_bags_train', type=int, default=128, metavar='NTrain',
                    help='number of bags in training set')
parser.add_argument('--num_bags_test', type=int, default=50, metavar='NTest',
                    help='number of bags in test set')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--model', type=str, default='attention', help='Choose b/w attention and gated_attention')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    print('\nGPU is ON!')

print('Load Train and Test Set')
loader_kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

source_dir = '/home/wenuka/Documents/Combined_Averaged/256'
train_files = '/home/wenuka/Documents/Combined_Averaged/final_train.txt'
log_path = 'train_sample_log_bag_shuffle11.txt'


train_loader = data_utils.DataLoader(LungDataset(source_dir,train_files),
                                     batch_size=5,
                                     collate_fn=custom_collate,
                                     **loader_kwargs)

test_loader = data_utils.DataLoader(MnistBags(target_number=args.target_number,
                                              mean_bag_length=args.mean_bag_length,
                                              var_bag_length=args.var_bag_length,
                                              num_bag=args.num_bags_test,
                                              seed=args.seed,
                                              train=False),
                                    batch_size=1,
                                    shuffle=False,
                                    **loader_kwargs)

print('Init Model')
if args.model=='attention':
    model = Attention()
elif args.model=='gated_attention':
    model = GatedAttention()
if args.cuda:
    model.cuda()

optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.reg)

class LossCallback:
    def __init__(self, patience=5, model=None):
        self.patience = patience
        self.model = model
        self.losses = []
        self.counter = 0
        self.best_loss = float('inf')
        self.stop_training = False

    def __call__(self, epoch, loss, error):
        self.losses.append(loss)
        if loss < self.best_loss:
            self.best_loss = loss
            self.counter = 0
            if self.model is not None:
                torch.save(self.model.state_dict(), "best_model_new_update_bag_shuffle.pth")
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"Loss hasn't decreased for {self.patience} consecutive epochs. Saving model.")
                self.stop_training = True


def reorder_batch(batch):
    new_array = []
    for i in range(len(batch['slide_name'])):
        label = batch['label'][i]
        features = batch['features'][i]
        for j in range(len(features)):
            new_array.append({label:features[j]})
    return new_array

def create_bag(index_list):
    bag_patches = []
    bag_features = []
    for k in index_list:
        for key, value in k.items():
            # Append key to indexes list
            bag_patches.append(key)
            # Extend elements list with values
            bag_features.append(value)
    random.shuffle(bag_features)
    return max(bag_patches),bag_features

def train(epoch,callback=None):
    model.train()
    numGroup = args.num_bags_train
    train_loss = 0.
    train_error = 0.
    for batch in train_loader:
        reordered_array = reorder_batch(batch) 
        index_chunk_list = np.array_split(np.array(reordered_array), numGroup)
        index_chunk_list = [sst.tolist() for sst in index_chunk_list]
        bag_probs = []
        bag_labels = []
        for tidx in range (len(index_chunk_list)):
            bag_label,bag_features = create_bag(index_chunk_list[tidx])
            bag_feat_tensor = [torch.tensor(f).to('cuda') for f in bag_features]
            bag_prob, _ = model.calculate_objective(torch.Tensor(bag_features).to('cuda'), torch.Tensor(bag_label).to('cuda'))
            bag_probs.append(bag_prob)
            bag_labels.append(bag_label)
        optimizer.zero_grad()
        bag_labels = torch.tensor(bag_labels, dtype=torch.int64).to('cuda')
        bag_preds = torch.cat(bag_probs,dim=0).to('cuda')   
        # neg_log_likelihood = -1. * (ground_truths * torch.log(bag_preds) + (1. - ground_truths) * torch.log(1. - bag_preds))  # negative log bernoulli
        neg_log_likelihood = F.nll_loss(bag_preds,bag_labels,reduction='mean')
        train_loss += neg_log_likelihood.item()
        # print(f"LOSS: {neg_log_likelihood.item()}")
        neg_log_likelihood.backward()
        optimizer.step()

    # calculate loss and error for epoch
    train_loss /= len(train_loader)
    train_error /= len(train_loader)

    log_line = 'Epoch: {}, Loss: {:.4f}, Train error: {:.4f}'.format(epoch, train_loss, train_error)
    print(log_line)
    print_log(log_line,log_file)

    if callback is not None:
        callback(epoch, train_loss, train_error)

def test():
    model.eval()
    test_loss = 0.
    test_error = 0.
    
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(test_loader):
            bag_label = label[0]
            instance_labels = label[1]
            if args.cuda:
                data, bag_label = data.cuda(), bag_label.cuda()
            data, bag_label = Variable(data), Variable(bag_label)
            print(f"X_Test Size: {data.shape} Y_test Size: {bag_label}")
            loss, attention_weights = model.calculate_objective(data, bag_label)
            test_loss += loss.data[0]
            error, predicted_label = model.calculate_classification_error(data, bag_label)
            test_error += error

            if batch_idx < 5:  # plot bag labels and instance labels for first 5 bags
                bag_level = (bag_label.cpu().data.numpy()[0], int(predicted_label.cpu().data.numpy()[0][0]))
                instance_level = list(zip(instance_labels.numpy()[0].tolist(),
                                    np.round(attention_weights.cpu().data.numpy()[0], decimals=3).tolist()))

                print('\nTrue Bag Label, Predicted Bag Label: {}\n'
                    'True Instance Labels, Attention Weights: {}'.format(bag_level, instance_level))

    test_error /= len(test_loader)
    test_loss /= len(test_loader)

    print('\nTest Set, Loss: {:.4f}, Test error: {:.4f}'.format(test_loss, test_error))

def print_log(tstr, f):
    f.write("\n")
    f.write(tstr)
    print(tstr)


if __name__ == "__main__":
    with open(log_path,'w') as f:
        f.write("model = {}\n".format(args.model))
        f.write("num_bags = {}\n".format(args.num_bags_train))
    callback = LossCallback(patience=10, model=model)
    print('Start Training')
    log_file = open(log_path,'a')
    for epoch in range(1, args.epochs + 1):
        train(epoch,callback)
        if callback.stop_training:
            print("Training stopped due to lack of improvement in loss.")
            break
    log_file.close()
