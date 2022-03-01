import torch
from torch import nn
from dataset_loader import BikeDataset
from torch.utils.data import DataLoader
from models import GRUmodel
import torch.optim as optim
import argparse
import numpy as np
import pandas as pd
import random
from metrics import Metric
from utils import save_results

def seed_everything(SEED=42):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True
SEED=42
seed_everything(SEED=SEED)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hourly Prediction of the Bike Dataset using PyTorch')

    # experiment setup
    parser.add_argument('-rate', default=0.001, type=float, help='learning rate')
    parser.add_argument('-batch_size', default=32, type=int, help='minibatch size')
    parser.add_argument('-epochs', default=200, type=int, help='train iters each timestep')
    parser.add_argument('-num_bins', default=20, type=int, help='train iters each timestep')

    # model
    parser.add_argument('-model', type=str, default='gru', help='only GRU is used for this file')
    parser.add_argument('-hidden_dim', type=int, default=10, help='hidden dimension in RNN hidden layer')
    parser.add_argument('-num_layers', type=int, default=1, help='the number of RNN  layers')
    parser.add_argument('-dropout', type=float, default=0.0, help='the probability for dropout [default: 0.0]')

    # Dataset
    parser.add_argument('-seqlen', type=int, default=1,
                        help='number of output classes in the generated training dataset')
    parser.add_argument('-prev_cnt', dest='prev_cnt', type=str, default='no',
                        help='Model(x_t, y_(t-i), y_(t-j): values: no, hour, week')
    parser.add_argument('-day_num', dest='day_num', type=int, default=1,
                        help='Model(x_t, y_(t-i), y_(t-j): values: no, hour, week')
    parser.add_argument('--reduced', dest='reduced', action='store_false',
                        help='if true, only uses the limited number of features which filtered by simple L1 regression')

    parser.set_defaults(reduced=False)
    args = parser.parse_args()

    num_bins = args.num_bins
    seq_len = args.seqlen

    # in case of arbitary bins, define the percent_bin and feed it to BikeDataset as an argument
    # percent_bins = [-1] + list(np.arange(-.75, .45, .05)) + list(np.arange(.5, 2.75, .25))
    # percent_bins.sort()

    # loading csv file
    dataset = pd.read_csv("dataset/hour.csv")

    training_set = BikeDataset(set_type='train', dataset=dataset, seq_len=seq_len, prev_cnt=args.prev_cnt,
                               reduced_features=args.reduced, num_bins=num_bins, day_num=args.day_num)

    # print(training_set.bins)
    print('Each input sample shape is ',training_set[0][0].shape)

    val_set = BikeDataset(set_type='val', dataset=dataset, seq_len=seq_len, prev_cnt=args.prev_cnt,
                          day_num=args.day_num,
                          reduced_features=args.reduced, percent_bins=training_set.bins, max_cnt=training_set.max_cnt,
                          repeated_data_num=training_set.repeated_data_num)

    test_set = BikeDataset(set_type='test', dataset=dataset, seq_len=seq_len, prev_cnt=args.prev_cnt,
                           day_num=args.day_num,
                           reduced_features=args.reduced, percent_bins=training_set.bins, max_cnt=training_set.max_cnt,
                           repeated_data_num=val_set.repeated_data_num)

    train_loader = DataLoader(training_set, batch_size=args.batch_size,
                              shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=len(val_set),
                            shuffle=False, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=len(test_set),
                             shuffle=False, drop_last=True)

    feature_size = training_set.feature_length
    if args.prev_cnt == 'no':
        previous_cnt_feature_size = 0
    elif args.prev_cnt == 'hour':
        previous_cnt_feature_size = 1
    elif args.prev_cnt == 'day':
        previous_cnt_feature_size = args.day_num

    model = GRUmodel(args, input_dim=feature_size + previous_cnt_feature_size,
                     val_test_batch=[len(val_set), len(test_set)], class_num=num_bins)

    # get the maximum value from the training as a reference to convert back normalize data to the orginal number
    model.max_val = training_set.max_cnt.numpy()
    optimizer = optim.Adam(model.parameters(), lr=args.rate)

    loss_function = nn.CrossEntropyLoss()

    data_loader = train_loader, val_loader, test_loader
    test_prev_out_continuous = test_set.prev_out_continuous
    test_out_continuous = test_set.out_continuous
    metric = Metric(train_loader, val_loader, test_loader, model, optimizer, loss_function,
                    training_set.bins, test_prev_out_continuous, test_out_continuous, model.max_val)

    train_loss_records = []
    val_loss_records = []
    test_loss_records = []

    val_std_records = []
    test_std_records = []


    mistake_counter = 0  # mistakes counter for validation loss
    for epoch in range(args.epochs):
        train_loss = metric.train()
        train_loss_records.append(train_loss)

        val_loss = metric.val_eval()
        val_loss_records.append(val_loss)

        if epoch > 20:
            if val_loss_records[-1] > val_loss_records[-2]:
                mistake_counter += 1

        test_loss, test_std = metric.test_eval()
        test_loss_records.append(test_loss)
        test_std_records.append(test_std)

        print(
            '[Epoch: %3d/%3d] Train CELoss: %.4f,    Val CELoss: %.4f,   Test MAE: %3.2f, STD: %3.2f'
            % (epoch, args.epochs, train_loss_records[epoch], val_loss_records[epoch], test_loss_records[epoch],
               test_std_records[epoch]))
        if mistake_counter > 10 or epoch == args.epochs - 1:
            print('TRAINING TERMINATED: validation loss has increased 10 times ')
            save_results(args=args, model=model, test_loss=test_loss, test_std=test_std, epoch=epoch)
            metric.graph()
            print(
                'Final Mean Absolute Error on test set:  Test MAE: %.4f, STD: %.4f'
                % ( test_loss_records[epoch], test_std_records[epoch]))
            break
