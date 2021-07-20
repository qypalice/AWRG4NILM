import torch
import numpy as np
import sys
from trainer import *
from datetime import datetime
import random
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from model import Conv2DAdaptiveRecurrence
from get_feature  import generate_input_feature
from dataset import *
import matplotlib.pyplot as plt

torch.set_default_tensor_type(torch.DoubleTensor)

def get_user_inputs():
    dataset_code = {'l': 'lilac', 'p': 'plaid'}
    dataset = dataset_code[input('Input l for lilac, p for plaid)')]
    image_code = {'a':'adaptive', 'v':'vi'}
    image_type = image_code[input('Input a for AWRG, v for VI-grapth')]
    eps = int(input('Input epsilon (suggest 10):'))
    delta = int(input('Input delta (suggest 10):'))
    width = int(input('Input width (suggest 50):'))
    return dataset, image_type, eps, delta, width


def create_trainer(dataset="lilac", image_type="adaptive", multi_dimension=True, batch_size=16,
            width = 50, eps=10, delta=10):
    # define parameters - input data
    in_size = 3 if dataset=="lilac" and multi_dimension==True else 1
    current, voltage, labels = get_data(submetered=False, data_type=dataset, isc=False)
    
    # get input features
    input_feature = generate_input_feature(current, voltage, image_type, width, multi_dimension)
    
    #perform label enconding
    le = LabelEncoder()
    le.fit(labels)
    y = le.transform(labels)

    # split the data
    skf = StratifiedKFold(n_splits=4,random_state=42,shuffle=True)
    train_index, test_index = next(skf.split(current, y))
    Xtrain, Xtest = input_feature[train_index], input_feature[test_index]
    ytrain, ytest = y[train_index], y[test_index]
    
    # define other parameters
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    loss_function = torch.nn.CrossEntropyLoss().to(device)
    classes=list(np.unique(ytrain))
    num_class=len(classes)
    train_loader, test_loader=get_loaders(Xtrain, Xtest, ytrain, ytest, batch_size=batch_size)
    model = Conv2DAdaptiveRecurrence(in_size=in_size, out_size=num_class,
                                            dropout=0.2, eps=eps, delta=delta, width=width)
    
    # create trainer
    trainer = Trainer(device, model, loss_function, train_loader, test_loader, 
                in_size=in_size, batch_size=batch_size, eps=eps, delta=10)
    print(f'Trainer for {dataset} created.', end = "\n")
    
    return num_class, trainer

def start_logging(filename):
    f = open('../logs/experiment-{}.txt'.format(filename), 'w')
    sys.stdout = f
    return f

def stop_logging(f):
    f.close()
    sys.stdout = sys.__stdout__

def train_the_model(trainer, dataset, image_type, epochs, width, multi_dimension=True):
    # define parameters
    file_name=f"{dataset}_{image_type}_{str(width)}"
    if dataset=="lilac" and multi_dimension==False :
        file_name = file_name+"_multi-dimension-norm"
    saved_model_path   = '../weight/{}_checkpoint.pt'.format(file_name)
    csv_logger = CSVLogger(filename=f'../logs/{file_name}.csv',
                       fieldnames=['epoch', 'train_loss', 'test_loss', 'train_acc', 'test_acc'])
    checkpoint = Checkpoint(saved_model_path, patience=100, checkpoint=True, score_mode="max",min_delta=1e-4)
    
    # initialize recording
    experiment_name = file_name+'_{}'.format(datetime.utcnow().strftime('%m-%d-%H-%M'))
    f = start_logging(experiment_name)
    print(f'Starting {experiment_name} experiment')

    # start training
    time_used, train_loss, train_acc,  test_loss, test_acc = trainer.train(epochs,  csv_logger, checkpoint, file_name)
    stop_logging(f)

    return checkpoint, file_name

def plot_learning_curve(file_name):
    # read data
    filename=f'../logs/{file_name}.csv'
    Data = np.loadtxt(open(filename),delimiter=",",skiprows=1)
    epoch = Data[:,0]
    train_loss = Data[:,1]
    train_acc = Data[:,2]
    test_loss = Data[:,3]
    test_acc = Data[:,4]
    
    # plot data
    labels = ['train_loss', 'test_loss', 'train_acc', 'test_acc']
    colors = ['r','y','g','b','m','k']
    fit =plt.figure()
    plt.plot(epoch, train_loss, color='r')
    plt.plot(epoch, train_acc, color='g')
    plt.plot(epoch, test_loss, color='k')
    plt.plot(epoch, test_acc, color='b')
    #plt.legend(labels=labels)
    plt.show() 

def test_the_model(trainer, num_class, checkpoint, file_name):
    f1, mcc, zl = trainer.test(num_class, checkpoint)
    filename = file_name+'_{}'.format(datetime.utcnow().strftime('%m-%d-%H-%M'))
    f = open('../results/{}.txt'.format(filename), 'w')
    sys.stdout = f
    print(f'macro-averaged F1 score: {str(f1)}.')
    print(f'Matthews correlation coefficient (MCC) score: {str(mcc)}.')
    print(f'Zero-loss score (ZL): {str(zl)}.')
    f.close()
    sys.stdout = sys.__stdout__
    
def fix_random_seed_as(random_seed):
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)

