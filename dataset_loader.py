import torch
from torch.utils.data import Dataset
import numpy as np
from utils import gen_equal_freq_bins

FLOAT = torch.FloatTensor
LONG = torch.LongTensor


class BikeDataset(Dataset):
    """
    Loading the Bike dataset using Pytorch Dataloader
    """

    def __init__(self, set_type, dataset, seq_len=1, prev_cnt='no', reduced_features=False,
                 percent_bins=None, num_bins=20, max_cnt=None, train_size=0.7,
                 validation_size=0.1, day_num=1, repeated_data_num=0):
        """
        :param timeframe: day or hour
        :param set_type: train, val, test
        :param seq_len: int and for LSTM models
        :param prev_cnt: if the cnt of the previous timestep is given as input. value: no (not given),
         hour (previous hour), week (day_num previous days and the same hour)
        :param reduced_features: full feature or the reduced ones which selected from L1 regression
        """

        self.seq_len = seq_len
        self.prev_cnt = prev_cnt
        self.repeated_data_num = 0
        self.reduced_features = reduced_features
        self.type = set_type
        self.repeated_data_num = repeated_data_num
        self.num_bins = num_bins
        data, start_idx = self.dataset_spliter(set_type, dataset, train_size, validation_size)

        x, y = self.data_normalizer(start_idx, data, max_cnt)
        self.raw_output = y
        # Printing number of the repeated samples, due to missing hours
        if self.prev_cnt == 'day':
            if set_type == 'train':
                print(
                    "Some hours are missing from the dataset. So, in case of missing sample, we repeat the next sample")
            print("number of repeated data in total is ", self.repeated_data_num)
            if set_type == 'test':
                print("==================================")

        self.input_output_preparation(x, y, percent_bins, seq_len, day_num)

    def input_output_preparation(self, x, y, percent_bins, seq_len, day_num):
        # Calculating the length of the (train/val/test) set
        # Also, calculating the shift of index for multi time steps


        # This part prepares the input, output pairs in a list of the torch tensor format
        # Later, it is used in the __getitem__  function to build the bataches.

        # Converting to categorical output
        if self.prev_cnt == 'no':
            rel_output_ref_hour_num = 24
            idx_shift = 24
        elif self.prev_cnt == 'hour' :
            rel_output_ref_hour_num = 1
            idx_shift = 1
        elif self.prev_cnt == 'day':
            rel_output_ref_hour_num = 24
            idx_shift = 24 * day_num

        self.set_len = len(x) - self.seq_len - idx_shift - 1

        self.output_preparation(y, idx_shift, percent_bins, rel_output_ref_hour_num)
        self.input_preparation(x, y, seq_len, day_num)



    def input_preparation(self, x, y, seq_len, day_num):
        # Multiple time steps and no additional info
        if self.prev_cnt == 'no':
            self.input = [torch.stack(x[idx + 1: idx + self.seq_len + 1]) for idx in range(self.set_len)]

        # Multiple time steps and the cnt of the previous hours for each time step is added to the input vector
        elif self.prev_cnt == 'hour':
            self.input = [torch.cat((torch.stack(x[idx + 1: idx + self.seq_len + 1]),
                                     torch.stack(y[idx: idx + self.seq_len]).unsqueeze(1)), dim=1)
                          for idx in range(self.set_len)]

        # Multiple time steps and the cnt of the same hours in the past day_num days of each time steps are added
        # as the input vector
        elif self.prev_cnt == 'day':
            self.input = torch.zeros((self.set_len, seq_len, day_num + self.feature_length))
            for idx in range(self.set_len):
                prev_days_same_hr = torch.zeros(self.seq_len, day_num)
                for seq in range(self.seq_len):
                    for day in range(day_num):
                        prev_days_same_hr[seq, day] = y[idx + 24 * (day + 1) + seq]
                self.input[idx, :, :] = torch.cat((torch.stack(x[idx: idx + self.seq_len]), prev_days_same_hr), dim=1)

    def output_preparation(self, y, idx_shift, percent_bins, hour_num):
        # calculating the relative change respect to the previous hour/day
        rel_inc = (np.array(y[idx_shift:]) - np.array(y[idx_shift - hour_num:-hour_num])) \
                  / (np.array(y[idx_shift - hour_num:-hour_num]) + 1e-10)

        if percent_bins is None:
            percent_bins = gen_equal_freq_bins(rel_inc, self.num_bins)

        self.percentage = rel_inc
        # converting relative change into discrete categories. The discretization is done based on the given bins.
        # the discrete output is from 1 to number of bins
        self.bins = np.array(percent_bins)
        output_discrete = np.digitize(rel_inc, bins=self.bins)

        # clipping the output in case of more than number of bins. Then minus 1 to start the classes numbers from 0
        output_categorical = np.clip(output_discrete, 1, len(self.bins) - 1) - 1
        assert 0 <= np.min(output_categorical) <= len(self.bins)

        # converting the numpy array to torch tensor
        self.output = torch.from_numpy(output_categorical).type(torch.long)

        # keeping the continuous output to calculate the MAE
        self.out_continuous = np.array(y[idx_shift:])
        self.prev_out_continuous = np.array(y[idx_shift - hour_num:-hour_num])

    def dataset_spliter(self, set_type, dataset, train_size, validation_size):
        '''
        This function splits the dataset into training, validation and test set given the percentage
        :param set_type: train, val or test
        :param dataset: the whole dataset
        :param train_size: size of the training set in terms fraction
        :param validation_size: size of the training set in terms fraction
        :return: the set, starting index
        '''
        len_dataset = len(dataset)

        # Train, val and test set separation
        trn_last_idx = int(train_size * len_dataset)
        val_last_idx = int((train_size + validation_size) * len_dataset)

        # validation and test set may not start from 00 hr due to a random splitting of dataset
        # we shift the index to start data from 00 hr
        # shift_index = 0

        # Splitting the dadaset
        if set_type == 'train':
            data = dataset[0:trn_last_idx]
            start_idx = 0

        elif set_type == 'val':

            # start the validation from a 00 hr sample
            start_idx = trn_last_idx
            for i in range(trn_last_idx + 1, trn_last_idx + 25):
                start_idx += 1
                # shift_index += 1
                if dataset['hr'][start_idx] == 0:
                    break
            data = dataset[start_idx: val_last_idx]

        elif set_type == 'test':

            # start the test set from a 00 hr sample
            start_idx = val_last_idx
            for i in range(val_last_idx + 1, val_last_idx + 25):
                start_idx += 1
                # shift_index += 1
                if dataset['hr'][start_idx] == 0:
                    break
            data = dataset[start_idx:]
        return data, start_idx


    def data_normalizer(self, start_idx, data, max_cnt):
        self.x_np = []
        self.y_np = []

        # preparing data from the start_idx of the dataset
        for idx in range(start_idx, start_idx + len(data)):

            # removing this index because many hours missing for those days
            if (571 <= idx <= 594) or (5618 <= idx <= 5651):
                continue

            # Some records are missing from the dataset, we repeat the next hour after those gaps
            if self.prev_cnt == 'day':

                # Checking if there is a missing hour
                number_of_repeat = 1 + data['hr'][idx] - (idx + self.repeated_data_num) % 24

                # keeping track of repeated samples to your
                self.repeated_data_num += (number_of_repeat - 1)

                assert number_of_repeat >= 0

            # if we do not include previous day(s) sample as input data then there is no need to repeat in case of
            # missing data
            else:
                number_of_repeat = 1

            # Repeating in case of missing hours
            for _ in range(number_of_repeat):

                # Dividing feature into important one and not important one based on L1 lin regression
                important_features = [
                    data['yr'][idx],
                    data['hr'][idx] / 23,
                    data['temp'][idx],
                    data['atemp'][idx],
                    data['hum'][idx],
                ]
                remaing_features = [
                    data['season'][idx] / 4,
                    data['mnth'][idx] / 12,
                    data['holiday'][idx],
                    data['weekday'][idx] / 6,
                    data['workingday'][idx],
                    data['weathersit'][idx] / 4,
                    data['windspeed'][idx]
                ]

                # Packing the list of features in a numpy array
                if self.reduced_features:
                    self.x_np.append(np.array(important_features))
                else:
                    self.x_np.append(np.array(important_features + remaing_features))
                self.y_np.append(np.array(data['cnt'][idx], dtype='f'))

        # print('idx=',idx,  ' rec_hr=', data['hr'][idx], ' id%24=',idx % 24, 'correct_hr=', round(self.x_np[-1][
        # 1]*23))

        self.feature_length = self.x_np[0].size

        # Converting to torch tensor
        y_raw = [torch.from_numpy(y) for y in self.y_np]
        x = [torch.from_numpy(x).type(FLOAT) for x in self.x_np]

        # Normalizing the output values based the maximum bike cnt in the training set
        if max_cnt is None:
            self.max_cnt = torch.stack(y_raw).max()
        else:
            self.max_cnt = max_cnt
        y = [val / self.max_cnt for val in y_raw]

        return x, y


    def __len__(self):
        return self.set_len

    def __getitem__(self, idx):
        return self.input[idx], self.output[idx]
