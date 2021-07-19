from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
from temp import *
from trainer import *
from functools import partial
from datetime import date
import random
import time
from functools import partial
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, f1_score, matthews_corrcoef, zero_one_loss
from model import Conv2DAdaptiveRecurrence, Conv2D
from visual_functions import plot_learning_curve, savefig
from get_feature  import generate_input_feature
from dataset import *


seed = 1054779
print("set seed")
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
else:
    torch.manual_seed(seed)



def train_model(Xtrain, Xtest, ytrain, ytest, width,
                 model_name, epochs=2, 
                 file_name=None,
                  in_size=1,batch_size=16, eps=10, delta=10):
    
    y_pred_total = []
    f_1_total = []
    mcc_total = []
    z_one_total = []
    y_test_total = []
    total_time = []
  
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    metric_fn = partial(get_accuracy)
    classes=list(np.unique(ytrain))
    num_class=len(classes)

    
    loaders=get_loaders(Xtrain, Xtest, ytrain, ytest, 
                        batch_size=batch_size)
    model = Conv2DAdaptiveRecurrence(in_size=in_size, out_size=num_class,
                                            dropout=0.2, eps=eps, delta=delta, width=width)

    weight_init(model)
    filename   = '{}_checkpoint.pt'.format(file_name)
    csv_logger = CSVLogger(filename=f'../logs/{file_name}.csv',
                       fieldnames=['epoch', 'train_loss', 'test_loss', 'train_acc', 'test_acc'])
    checkpoint = Checkpoint(filename, patience=100, checkpoint=True, score_mode="max",min_delta=1e-4)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, nesterov=True)
    
    start_time =  time.time()

    train_loss, train_acc,  test_loss, test_acc = perform_training(epochs, 
                                                        model, 
                                                        loaders['train'],
                                                        loaders['val'],
                                                        optimizer, 
                                                        criterion, 
                                                        device, 
                                                        checkpoint,
                                                        metric_fn,
                                                        csv_logger)
    end_time    = time.time()
    time_used   = end_time - start_time
    pred, test  = get_prediction(model, loaders['val'], checkpoint, device,1)
    plot_learning_curve(train_loss, train_acc, test_loss, test_acc)
    savefig(f"../figure/temp/{file_name}",format=".pdf")
    
    f1 = f1_score(test, pred,  average='weighted')
    mcc  = matthews_corrcoef(test, pred)
    zl   = zero_one_loss(test, pred)*100
    f_1_total.append(f1)
    mcc_total.append(mcc)
    z_one_total.append(zl)
    y_pred_total.append(pred.astype(int))
    y_test_total.append(test.astype(int))
    total_time.append(time_used)

    np.save(f"../results/{file_name}_pred.npy", np.vstack(y_pred_total))
    np.save(f"../results/{file_name}_f1.npy", np.vstack(f_1_total))
    np.save(f"../results/{file_name}_mcc.npy", np.vstack(mcc_total))
    np.save(f"../results/{file_name}_z_one.npy", np.vstack(z_one_total))
    np.save(f"../results/{file_name}_true.npy", np.vstack(y_test_total))
    np.save(f"../results/{file_name}_time.npy", np.vstack(total_time))



def parameters_initilisation_experiments(dataset="lilac",
                            image_type="adaptive",
                            width=50,
                            run_id=1,
                            epochs=100,
                            multi_dimension=True,
                            batch_size=16,
                            model_name="CNN"):

    in_size = 3 if dataset=="lilac" and multi_dimension==True else 1
    current, voltage, labels = get_data(submetered=False, data_type=dataset, isc=False)
    #create input feature representation
    
    
    
    #perform label enconding
    le = LabelEncoder()
    le.fit(labels)
    y = le.transform(labels)

    
    skf = StratifiedKFold(n_splits=4,random_state=42,shuffle=True)
    train_index, test_index = next(skf.split(current, y))
    
    input_feature = generate_input_feature(current, voltage, image_type, width, multi_dimension)
    Xtrain, Xtest = input_feature[train_index], input_feature[test_index]
    ytrain, ytest = y[train_index], y[test_index]

    for delta in [0, 1, 5, 10, 20, width]:
        for eps in [0, 1, 10, 20, 30, 40, 50]:
            file_name=f"{dataset}_{image_type}_{str(width)}_{model_name}_{str(run_id)}_parameters_{str(delta)}_{str(eps)}"
            if dataset=="lilac" and multi_dimension==False :
                file_name = file_name+"_multi-dimension-norm"
            print(f"fit model for {file_name}")
            train_model(Xtrain, Xtest, ytrain, ytest, width,
                    model_name, epochs, file_name, in_size,batch_size, delta, eps)


def embending_size_parameter_experiments(dataset="lilac",
                            image_type="adaptive",
                            run_id=1,
                            epochs=100,
                            multi_dimension=True,
                            batch_size=16,
                            model_name="CNN"):

    in_size = 3 if dataset=="lilac" and multi_dimension==True else 1
    current, voltage, labels = get_data(submetered=False, data_type=dataset, isc=False)
    #create input feature representation
    
    
    
    #perform label enconding
    le = LabelEncoder()
    le.fit(labels)
    y = le.transform(labels)

    
    skf = StratifiedKFold(n_splits=4,random_state=42,shuffle=True)
    train_index, test_index = next(skf.split(current, y))
    
    eps=10
    delta=10
    for  width in [30, 50, 60, 80, 100]:
        file_name=f"{dataset}_{image_type}_{str(width)}_{model_name}_{str(run_id)}_parameters_emb_size_{str(delta)}_{str(eps)}"
        input_feature = generate_input_feature(current, voltage, image_type, width, multi_dimension)
        Xtrain, Xtest = input_feature[train_index], input_feature[test_index]
        ytrain, ytest = y[train_index], y[test_index]
        if dataset=="lilac" and multi_dimension==False :
            file_name = file_name+"_multi-dimension-norm"
        print(f"fit model for {file_name}")
        train_model(Xtrain, Xtest, ytrain, ytest, width,
                    model_name, epochs, file_name, in_size,batch_size, delta, eps)

def start_logging(filename):
    f = open('../logs/experiment-{}.txt'.format(filename), 'w')
    sys.stdout = f
    return f

def stop_logging(f):
    f.close()
    sys.stdout = sys.__stdout__

if __name__ == '__main__':
    #DEFINE EXPERIMENTAL VARIABLES for EXP ONE 
    __spec__ = None
    datasets = ["plaid", "lilac"]
    feature = "adaptive"
    n_epochs  =  100
    batch_size = 16
    width=50


    #RUN EXPERIMENT ONE_EMBENDING
    experiment_name = 'ADRP-NILM-experiment_one_emb_size_parameter_{}'.format(date.today().strftime('%m-%d-%H-%M'))
    f = start_logging(experiment_name)
    print(f"Starting {experiment_name} experiment")

    for dataset in datasets:
        embending_size_parameter_experiments(dataset=dataset,
                                    image_type = feature,
                                    epochs = n_epochs,
                                    batch_size = batch_size)
    stop_logging(f)
    exit()


    #DEFINE EXPERIMENTAL VARIABLES for EXP ONE 
    datasets = ["lilac", "plaid"]
    features = ["vi","adaptive"]
    n_epochs  =  300
    batch_size = 16
    width=50

    #RUN EXPERIMENT TWO
    experiment_name = 'ADRP-NILM-experiment_two_comparison_with_baseline:{}'.format(date.today().strftime('%m-%d-%H-%M'))
    f = start_logging(experiment_name)
    print(f"Starting {experiment_name} experiment")
    for dataset in datasets:
        for feature in features:
            comparison_with_baseline_experiments(dataset=dataset,
                                        image_type = feature ,
                                        width  = width,
                                        epochs = n_epochs,
                                        batch_size = batch_size)
    stop_logging(f)
    exit()