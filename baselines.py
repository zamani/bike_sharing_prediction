from sklearn import linear_model
from sklearn.metrics import mean_absolute_error as mae
from dataset_loader import BikeDataset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils import gen_equal_freq_bins, output_normalization


if __name__ == '__main__':

    # loading csv file
    dataset = pd.read_csv("dataset/hour.csv")

    # Splitting the dataset into training, validation, and test set
    # The split can change by train_size and validation_size keywords
    # The default split are train_size=0.7 and validation_size=0.1
    traing_set = BikeDataset('train', dataset=dataset)
    val_set = BikeDataset( 'val', dataset=dataset)
    test_set = BikeDataset( 'test', dataset=dataset)

    #Linear Regression Model
    reg = linear_model.LinearRegression()
    reg.fit(traing_set.x_np,traing_set.y_np)
    print('Various baselines are demonstrated in this code')
    print('Linear Regression without regularization')

    prediction = reg.predict(val_set.x_np)
    target = val_set.y_np
    print('Validation set loss:  {:.2f}, std: {:.2f}'
          .format(mae(prediction, target), np.std(abs(prediction-target))))

    prediction = reg.predict(test_set.x_np)
    target = test_set.y_np
    print('Test set loss: {:.2f}, std: {:.2f}'
          .format(mae(prediction, target), np.std(abs(prediction-target))))
    print('================================================================')


    reg = linear_model.Ridge(alpha=.5)
    reg.fit(traing_set.x_np,traing_set.y_np)
    print('Linear Regression with Ridge regularization')

    prediction = reg.predict(val_set.x_np)
    target = val_set.y_np
    print('Validation set loss: {:.2f}, std: {:.2f}'
          .format(mae(prediction, target), np.std(abs(prediction-target))))

    prediction = reg.predict(test_set.x_np)
    target = test_set.y_np
    print('Test set loss: {:.2f}, std: {:.2f}'
          .format(mae(prediction, target), np.std(abs(prediction-target))))
    print('================================================================')

    reg = linear_model.Lasso(alpha=.1)
    reg.fit(traing_set.x_np,traing_set.y_np)
    print('Linear Regression with Lasso regularization')

    prediction = reg.predict(val_set.x_np)
    target = val_set.y_np
    print('Validation set loss: {:.2f}, std: {:.2f}'
          .format(mae(prediction, target), np.std(abs(prediction-target))))

    prediction = reg.predict(test_set.x_np)
    target = test_set.y_np
    print('Test set loss: {:.2f}, std: {:.2f}'
          .format(mae(prediction, target), np.std(abs(prediction-target))))
    print('================================================================')

    reg = linear_model.LassoLars(alpha=.1)
    reg.fit(traing_set.x_np,traing_set.y_np)

    prediction = reg.predict(val_set.x_np)
    target =  val_set.y_np
    print('validation set loss with Lasso LARS regularization {:.2f}, std: {:.2f}'
          .format(mae(prediction, target), np.std(abs(prediction-target))))

    prediction = reg.predict(test_set.x_np)
    target = test_set.y_np
    print('test set loss with Lasso LARS regularization {:.2f}, std: {:.2f}'
          .format(mae(prediction, target), np.std(abs(prediction-target))))
    #print(reg.coef_)
    print('================================================================')

    print('Using prvious cnt number as prediction for the current cnt')

    prediction = np.stack(traing_set.y_np[1:])
    target = np.stack(traing_set.y_np[:-1])
    print('train error: {:.2f}, std: {:.2f}'
          .format(mae(prediction, target), np.std(abs(prediction-target))))

    prediction = np.stack(val_set.y_np[1:])
    target = np.stack(val_set.y_np[:-1])
    print('val error  : {:.2f}, std: {:.2f}'
          .format(mae(prediction, target), np.std(abs(prediction-target))))

    prediction = np.stack(test_set.y_np[1:])
    target = np.stack(test_set.y_np[:-1])
    print('test error : {:.2f}, std: {:.2f}'
          .format(mae(prediction, target), np.std(abs(prediction-target))))
    print('================================================================')

    print('Using the change rate from cnt(t-2) to cnt(t-1) to predict the currrent cnt')
    prediction = np.stack(traing_set.y_np[2:])
    target = 2 * np.stack(traing_set.y_np[1:-1]) - np.stack(traing_set.y_np[0:-2])
    print('train error: {:.2f}, std: {:.2f}'
          .format(mae(prediction, target), np.std(abs(prediction-target))))

    prediction = np.stack(val_set.y_np[2:])
    target = 2 * np.stack(val_set.y_np[1:-1])- np.stack(val_set.y_np[0:-2])
    print('val error  : {:.2f}, std: {:.2f}'
          .format(mae(prediction, target), np.std(abs(prediction-target))))

    prediction = np.stack(test_set.y_np[2:])
    target = 2 * np.stack(test_set.y_np[1:-1]) - np.stack(test_set.y_np[0:-2])
    print('test error : {:.2f}, std: {:.2f}'
          .format(mae(prediction, target), np.std(abs(prediction-target))))
    print('================================================================')

    print('Using prvious day same hour cnt number as prediction for the current cnt')

    prediction = np.stack(traing_set.y_np[24:])
    target = np.stack(traing_set.y_np[:-24])
    print('train error: {:.2f}, std: {:.2f}'
          .format(mae(prediction, target), np.std(abs(prediction-target))))

    prediction = np.stack(val_set.y_np[24:])
    target = np.stack(val_set.y_np[:-24])
    print('val error  : {:.2f}, std: {:.2f}'
          .format(mae(prediction, target), np.std(abs(prediction-target))))

    prediction = np.stack(test_set.y_np[24:])
    target = np.stack(test_set.y_np[:-24])
    print('test error : {:.2f}, std: {:.2f}'
          .format(mae(prediction, target), np.std(abs(prediction-target))))
    print('================================================================')

    #Showing the histogram of relative change in number of rented bikes with respect to previous hour
    #percent_bins = [-1 + i/10 for i in range(31)] # simpler bin file
    percent_bins = [-1] + list(np.arange(-.75, .45, .05)) + list(np.arange(.5, 2.75, .25))
    print('Histogram of rate change compared to previous hour of the following bins are shown:')
    print('Percentage bins: ',['%.2f' % i for i in percent_bins])
    fig, axes = plt.subplots(2, 1)
    traing_set = BikeDataset('train', dataset=dataset, seq_len=2, prev_cnt='hour', percent_bins=percent_bins)
    axes[0].hist(traing_set.percentage, bins=percent_bins, edgecolor='black', linewidth=1.2)
    axes[0].set_title('relative change in number of rental bikes with respect to previous hour with arbitary bins')
    axes[0].set_ylabel('Bike rental count')
    
    # histogram of relative change in number of rental bikes with Equal Frequency Binning
    rel_outputs = output_normalization(traing_set.raw_output, 1, 1)
    percent_bins_equal_freq = gen_equal_freq_bins(rel_outputs, 20)
    traing_set = BikeDataset('train', dataset=dataset, seq_len=2, prev_cnt='hour', percent_bins=percent_bins_equal_freq)
    axes[1].hist(traing_set.percentage, bins=percent_bins_equal_freq, edgecolor='black', linewidth=1.2)
    axes[1].set_title('relative change in number of rental bikes with respect to previous hour with Equal Frequency Binning')
    axes[1].set_ylabel('Bike rental count')
    plt.show()