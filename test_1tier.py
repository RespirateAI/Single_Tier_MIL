from __future__ import print_function

import argparse

import numpy as np
import torch
import torch.optim as optim
import torch.utils.data as data_utils
from torch.autograd import Variable

from dataloader import MnistBags
from CT_dataloader import custom_collate,LungDataset

from model_CT import Attention, GatedAttention


# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST bags Example')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
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
parser.add_argument('--num_bags_train', type=int, default=1, metavar='NTrain',
                    help='number of bags in training set')
parser.add_argument('--num_bags_test', type=int, default=1, metavar='NTest',
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

source_dir = '/home/wenuka/Documents/NSCLC/256'
train_files = '/home/wenuka/Documents/Combined_Averaged/final_train.txt'
test_files =  '/home/wenuka/Documents/Combined_Averaged/final_test.txt'
log_path = 'train_sample_log_update3_Image_Per_Bag.txt'
attention_path = '/media/wenuka/New Volume-G/01.FYP/Tool_1/Simple MIL/AttentionDeepMIL/attention/attention_1image_par_bag.npy'



test_loader = data_utils.DataLoader(LungDataset(source_dir,test_files),
                                     batch_size=1,
                                     collate_fn=custom_collate,
                                     **loader_kwargs)

print('Init Model')
if args.model=='attention':
    model = Attention()
elif args.model=='gated_attention':
    model = GatedAttention()
if args.cuda:
    model.cuda()

tsave_dict = torch.load('best_model_Image_Per_Bag.txt.pth')
model.load_state_dict(tsave_dict)


optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.reg)

def print_log(tstr, f):
    f.write("\n")
    f.write(tstr)
    print(tstr)

def test():
    attention_dict = {}
    model.eval()
    test_loss = 0.
    test_error = 0.
    
    with torch.no_grad():
        for (id,batch) in enumerate(test_loader):
            slide_name_batch = batch["slide_name"][0]
            features_batch = batch["features"]
            labels_batch = batch["label"]

            label_tensor = torch.LongTensor(labels_batch).to('cuda')
            tslideLabel = label_tensor[0].unsqueeze(0)
            batch_feat = [torch.tensor(f).to('cuda') for f in features_batch]
            
            loss, attention_weights = model.calculate_objective(batch_feat[0], tslideLabel)
            test_loss += loss.data[0]
            error, predicted_label = model.calculate_classification_error(batch_feat[0], tslideLabel)
            test_error += error
            attention_dict[slide_name_batch] = attention_weights
            
    test_error /= len(test_loader)
    test_loss /= len(test_loader)

    print('\nTest Set, Loss: {:.4f}, Test error: {:.4f}'.format(test_loss.cpu().numpy()[0], test_error))
    np.save(attention_path,attention_dict)


if __name__ == "__main__":

    print('Start Testing')
    test()
    
