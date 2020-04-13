import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import matplotlib.pyplot as plt
import time
import numpy as np
import imageio
import sys

torch.manual_seed(1)  # reproducible




# this is one way to define a network
class NeuralNet(torch.nn.Module):
    def __init__(self, input_size, hidden_layer_size1, hidden_layer2_size, output_size):
        super(NeuralNet, self).__init__()
        if hidden_layer2_size <= 0:
            self.net = nn.Sequential(
                torch.nn.Linear(input_size, hidden_layer_size1),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(hidden_layer_size1, 1)
            )
        else:
            self.net = nn.Sequential(
                torch.nn.Linear(input_size, hidden_layer_size1),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(hidden_layer_size1, hidden_layer2_size),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(hidden_layer2_size, 1)
            )

    def forward(self, x):
        x = self.net(x)
        return x

def create_and_show_data(dataset_name, dataset_size, level_of_noise):
    x = torch.unsqueeze(torch.linspace(-np.pi, np.pi, dataset_size), dim=1)  # x data (tensor), shape=(100, 1)
    if dataset_name == 'sine':
        y = x.sin() + level_of_noise * torch.rand(x.size()) - level_of_noise/2   # noisy y data (tensor), shape=(100, 1)
        y_target = x.sin()
    elif dataset_name == 'cuadratic':
        y = 0.2 * x.pow(2) - 1 + level_of_noise * torch.rand(x.size()) - level_of_noise / 2
        y_target = 0.2 * x.pow(2) - 1
    else:
        y = x/3 + level_of_noise * torch.rand(x.size()) - level_of_noise / 2
        y_target = x/3
        # torch can only train on Variable, so convert them to Variable
    x, y, y_target = Variable(x), Variable(y), Variable(y_target)

    # # view data
    # plt.figure(figsize=(4, 4))
    # plt.scatter(x.data.numpy(), y.data.numpy(), color="orange", label='Training data')
    # plt.plot(x.data.numpy(), y_target.data.numpy(), color="red", label='Target function')
    # plt.title('Regression Analysis')
    # plt.xlabel('Independent varible')
    # plt.ylabel('Dependent varible')
    # plt.legend()
    # plt.show()
    return x, y, y_target



def save_nn_output(x, prediction, y, y_target, loss, epoch):
    fig, ax = plt.subplots(figsize=(12, 7))
    plt.cla()
    ax.set_title('Regression Analysis', fontsize=35)
    ax.set_xlabel('Independent variable', fontsize=24)
    ax.set_ylabel('Dependent variable', fontsize=24)
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-1.25, 1.25)
    ax.scatter(x.data.numpy(), y.data.numpy(), color="orange")
    plt.plot(x.data.numpy(),  y_target.data.numpy(), color="red", label='Target function')
    ax.plot(x.data.numpy(), prediction.data.numpy(), 'g-', lw=3)
    ax.text(1.0, 0.1, 'Step = %d' % epoch, fontdict={'size': 24, 'color': 'red'})
    ax.text(1.0, -0.1, 'Loss = %.4f' % loss.data.numpy(),
            fontdict={'size': 24, 'color': 'red'})

    # Used to return the plot as an image array
    # (https://ndres.me/post/matplotlib-animated-gifs-easily/)
    fig.canvas.draw()  # draw the canvas, cache the renderer
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return image, plt


def save_loss_output(loss_values, training_epochs):
    fig, ax = plt.subplots(figsize=(12, 7))
    plt.cla()
    ax.set_title('Regression Analysis - Loss', fontsize=35)
    ax.set_xlabel('Training epoch', fontsize=24)
    ax.set_ylabel('Loss', fontsize=24)
    ax.set_xlim(0, training_epochs)
    ax.set_ylim(0, 1.0)
    ax.plot(np.array(list(range(len(loss_values)))), np.array(loss_values), color="blue")
    ax.text(int(training_epochs/2), 0.5, 'Loss = %.4f' % loss_values[-1].data.numpy(),
            fontdict={'size': 24, 'color': 'red'})

    # Used to return the plot as an image array
    fig.canvas.draw()  # draw the canvas, cache the renderer
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return image, plt

def show_user_help(dataset_names):
    print('Check the use of this program.')
    print('python 01-nn-regression.py <dataset name> <dataset size> <level of noise> <hidden layer 1 size> <hidden layer 2 size> <training epochs> <learning rate>')
    print('\t<datasel name>:\tselect from {}'.format(dataset_names))
    sys.exit(-1)

def control_input_errors(argv):
    dataset_names = ['linear', 'cuadratic', 'sine']
    error = False
    if len(argv) != 8:
        print('ERROR: Number of parameters error')
        show_user_help(dataset_names)
    if not argv[1] in dataset_names:
        print('ERROR: dataset_name should be one of the following ones: {}'.format(dataset_names))
        error = True
    if int(argv[4]) <= 0:
        print('ERROR: Actual parameter hidden layer 1 size: {}, but it should be > 0. '.format(argv[4]))
        error = True
    if float(argv[6]) <= 0:
        print('ERROR: Actual parameter learning rate: {}, but it should be > 0. '.format(argv[6]))
        error = True
    if int(argv[7]) <= 0:
        print('ERROR: Actual parameter training epochs: {}, but it should be > 0. '.format(argv[7]))
        error = True
    if error: show_user_help(dataset_names)

def main():
    control_input_errors(sys.argv)
    show_training = False

    dataset_name = sys.argv[1]
    dataset_size = int(sys.argv[2])
    level_of_noise = float(sys.argv[3])
    x, y, y_target = create_and_show_data(dataset_name, dataset_size, level_of_noise)

    hidden_layer1_size = int(sys.argv[4])
    hidden_layer2_size = 0 if int(sys.argv[5]) < 0 else int(sys.argv[5])
    net = NeuralNet(1, hidden_layer1_size, hidden_layer2_size, 1)

    learning_rate = float(sys.argv[6])
    training_epochs = int(sys.argv[7])
    optimizer = optim.SGD(net.parameters(), lr=learning_rate)
    loss_functon = nn.MSELoss()  # this is for regression mean squared loss

    print('Experiment configuration:')
    print('Dataset name: {},\tDataset size: {},\tLebel of noise: {}, Hidden layer 1 size: {},\tHidden layer 2 size: {}\tLearning rate: {},\tTraning epochs: {}'
          .format(dataset_name, dataset_size, level_of_noise, hidden_layer1_size, hidden_layer2_size, learning_rate, training_epochs))

    output_images = []
    loss_images = []
    loss_values = []

    # train the network
    start = time.time()
    for epoch in range(training_epochs):
        prediction = net(x)  # input x and predict based on x

        loss = loss_functon(prediction, y)  # must be (1. nn output, 2. target)
        loss_values.append(loss)

        optimizer.zero_grad()  # clear gradients for next train
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients

        if show_training: print('Epoch={}\tLoss={}'.format(epoch, loss))
        # plot and show learning process
        output_images.append(save_nn_output(x, prediction, y, y_target, loss, epoch)[0])
        loss_images.append(save_loss_output(loss_values, training_epochs)[0])

    elapsed_time = time.time() - start
    print('Final loss={}, Computation time={} seconds.'.format(loss_values[-1], elapsed_time))
    if show_training: print('Creating gifs...')

    save_nn_output(x, prediction, y, y_target, loss, training_epochs)[1].savefig(
        './output/network_output-{}-{}-{}_{}epochs_nn-{}-{}.png'.format(dataset_name, dataset_size,
                                                                        level_of_noise, training_epochs,
                                                                        hidden_layer1_size, hidden_layer1_size))
    save_loss_output(loss_values, training_epochs)[1].savefig(
        './output/loss_output-{}-{}-{}_{}epochs_nn-{}-{}.png'.format(dataset_name, dataset_size,
                                                                     level_of_noise, training_epochs,
                                                                     hidden_layer1_size, hidden_layer1_size))

    # save images as a gif
    imageio.mimsave('./output/network_output-{}-{}-{}_{}epochs_nn-{}-{}.gif'.format(dataset_name, dataset_size,
                                                                                    level_of_noise, training_epochs,
                                                                                    learning_rate,
                                                                                    hidden_layer1_size,
                                                                                    hidden_layer1_size), output_images,
                    fps=10)
    imageio.mimsave('./output/loss_output-{}-{}-{}_{}epochs_lr{}_nn-{}-{}.gif'.format(dataset_name, dataset_size,
                                                                                 level_of_noise, training_epochs,
                                                                                 learning_rate,
                                                                                 hidden_layer1_size,
                                                                                 hidden_layer1_size), loss_images,
                    fps=10)

main()
