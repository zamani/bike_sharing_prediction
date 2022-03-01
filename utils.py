import datetime
import os
import torch
import numpy as np


def save_results(args, model, test_loss, test_std, epoch):
    data = {
        "model": model,
        "opt": args,
        "summary": test_loss
    }
    date = datetime.datetime.now().strftime("%I_%M_%S_%f_%p_on_%B_%d_%Y")
    model_name = "bike_sharing__TestMAE={metrics:.2f}__STD={std:.2f}__ep={epoch_id}__model={model_name}" \
        .format(metrics=test_loss, std=test_std, epoch_id=epoch, model_name=args.model)
    shared_string = '__lyr=  ' + str(args.num_layers) + '__hidNum=' + str(args.hidden_dim)
    dateset_info = '__SeqLen=' + str(args.seqlen) + '__Prev-cnt=' + str(args.prev_cnt) + '__Reduced-features=' + str(
        args.reduced)
    subfolder = 'results/' + model_name + shared_string + dateset_info + '__TIME=' + date
    if not os.path.exists(subfolder):
        os.makedirs(subfolder)

        filename = subfolder + '/' + 'model'

    torch.save(data, filename)


def gen_equal_freq_bins(x, nbin):
    nlen = len(x)
    return np.interp(np.linspace(0, nlen, nbin + 1),
                     np.arange(nlen),
                     np.sort(x))


def output_normalization(y, idx_shift, hour_num):
    # calculating the relative change respect to the previous hour/day
    rel_inc = (np.array(y[idx_shift:]) - np.array(y[idx_shift - hour_num:-hour_num])) \
              / (np.array(y[idx_shift - hour_num:-hour_num]) + 1e-5)
    return rel_inc
