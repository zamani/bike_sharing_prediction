import torch
from torch import nn

class GRUmodel(nn.Module):
    def __init__(self, args, input_dim, val_test_batch, class_num):
        super(GRUmodel, self).__init__()

        self.args = args
        self.hidden_dim = args.hidden_dim
        self.num_layers = args.num_layers
        self.batch_size = args.batch_size

        # For the validation and test the whole set is taken at once, so we change the batch size
        self.inner_batch_size = args.batch_size
        self.val_test_batch = val_test_batch


        self.gru = nn.GRU(input_size=input_dim, hidden_size=self.hidden_dim, dropout=args.dropout,
                            num_layers=self.num_layers, batch_first=True)

        self.init_hidden(mode='train')
        self.relu = nn.ReLU()
        self.lin1 = nn.Linear(self.hidden_dim , 64)
        self.lin2 = nn.Linear(64, class_num)

    def init_hidden(self, mode):

        # changing the batch size in case of the validation, test and final graph
        if mode == 'train':
            batch_size = self.batch_size
        elif mode == 'val':
            batch_size = self.val_test_batch[0]
        elif mode == 'test' or mode == 'graph':
            batch_size = self.val_test_batch[1]

        self.inner_batch_size = batch_size

        self.hidden = torch.zeros( self.num_layers, batch_size, self.hidden_dim, requires_grad=True) # h

    def forward(self, input_batch):

        # feeding input to batch
        lstm_out, last_hidden = self.gru(input_batch, self.hidden)

        # taking last hidden state to the fully connected layer
        out = self.lin1(last_hidden[-1])

        # and a fully connected layer
        out = self.relu(out)
        out = self.lin2(out)
        return out
