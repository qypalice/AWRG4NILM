import torch
import torch.nn as nn
import torch.nn.init as init
from tqdm import tqdm
import os
import sys
import time
import csv
import numpy as np
from abc import *
from dataset import *
from sklearn.metrics import f1_score, matthews_corrcoef, zero_one_loss
from sklearn.preprocessing import MultiLabelBinarizer

torch.set_default_tensor_type(torch.DoubleTensor)

class Checkpoint(object):
    
    def __init__(self,filename: str=None, patience: int =20, checkpoint: bool = False, score_mode: str="max",
                 min_delta: float=1e-4, save_final_model: bool = False):

        self.saved_model_path = '../checkpoint/'+filename
        self.checkpoint = checkpoint
        self.save_final_model = save_final_model
        self.patience = patience
        self.min_delta = min_delta
        self.score_mode = score_mode
        self.num_bad_epochs = 0
        self.is_better = None
        self.best = None
        self._init_is_better(score_mode, min_delta)
        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

        

    def _init_is_better(self, score_mode, min_delta):
        if score_mode not in {'min', 'max'}:
            raise ValueError('mode ' + score_mode + ' is unknown!')

        elif score_mode == 'min':
            self.is_better = lambda a, best: a < best - min_delta

        elif score_mode == 'max':
            self.is_better = lambda a, best: a > best + min_delta

    def early_stopping(self, metric, states):

        if self.best is None:
            self.best = metric

        if self.is_better(metric, self.best):
            self.num_bad_epochs = 0
            self.best = metric
            states['best_score'] = self.best

            if self.checkpoint:
                self.save_checkpoint(states)

        else:
            self.num_bad_epochs += 1

        if (self.num_bad_epochs >= self.patience) or np.isnan(metric):
            terminate_flag = True

        else:
            terminate_flag = False

        return terminate_flag

    def save_checkpoint(self, state):
        """
        Save best models
        arg:
           state: model states
           is_best: boolen flag to indicate whether the model is the best model or not
           saved_model_path: path to save the best model.
        """
        print("save best model")
        torch.save(state['state_dict'], self.saved_model_path)

        #torch.save(state, self.saved_model_path)

    def load_saved_model(self, model):
        saved_model_path = self.saved_model_path

        if os.path.isfile(saved_model_path):
            model.load_state_dict(torch.load(saved_model_path))
        else:
            print("=> no checkpoint found at '{}'".format(saved_model_path))
            
        return model

class CSVLogger():
    def __init__(self, filename, fieldnames=['epoch']):

        self.filename = filename
        self.csv_file = open(filename, 'w')

        # Write model configuration at top of csv
        """
        writer = csv.writer(self.csv_file)
        
        for arg in vars(args):
            writer.writerow([arg, getattr(args, arg)])
        writer.writerow([''])
        """

        self.writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
        self.writer.writeheader()

        self.csv_file.flush()

    def writerow(self, row):
        self.writer.writerow(row)
        self.csv_file.flush()

    def close(self):
        self.csv_file.close()

class Trainer(metaclass=ABCMeta):
    def __init__(self, device, model, loss_function, 
                train_loader, test_loader, 
                in_size=1, batch_size=16, eps=10, delta=10):    
        # import param
        self.device = device
        self.model = model.to(device)
        self.loss_function = loss_function
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.in_size = in_size
        self.batch_size = batch_size
        self.eps = eps
        self.delta = delta
        
        # generate param
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3, momentum=0.9, nesterov=True)

    def train(self,  epochs,  csv_logger, checkpoint, file_name):
        # initialize parameters
        self._weight_init(self.model)

        # start training
        start_time =  time.time()
        train_loss = []
        train_acc  = []
        test_loss  = []
        test_acc   = []        
        
        for epoch in range(epochs):
            loss_tra, score_tra = self.train_one_epoch(epoch)
            loss_tra, score_tra  = self.validate(self.train_loader)
            train_loss.append(loss_tra)
            train_acc.append(score_tra)
            
            loss_test, score_test  = self.validate(self.test_loader)
            test_loss.append(loss_test)
            test_acc.append(score_test)
            
            tqdm.write('test_loss: %.3f, test_score: %.4f' % (loss_test, score_test))
            states = {
                        'epoch': epoch+1,
                        'state_dict': self.model.state_dict()
                    }

            row = {'epoch': str(epoch), 'train_loss': str(loss_tra), 'test_loss': str(loss_test),
                    'train_acc': str(score_tra), 'test_acc': str(score_test)}
            
            csv_logger.writerow(row)
            if checkpoint.early_stopping(score_test, states) and (epoch+1)>=int(epochs*0.5):
                tqdm.write("Early stopping with {:.3f} best score, the model did not improve after {} iterations".format(
                        checkpoint.best, checkpoint.num_bad_epochs))
                break
            
        csv_logger.close()
        end_time    = time.time()
        time_used   = end_time - start_time

        # record training process
        #plot_learning_curve(train_loss, train_acc, test_loss, test_acc)
        #savefig(f"../figure{file_name}",format=".pdf")
        return time_used, train_loss, train_acc,  test_loss, test_acc

    def test(self, num_class, checkpoint):
        # get prediction
        pred, test  = self._get_prediction(checkpoint, self.test_loader, num_class)

        # get scores
        m = MultiLabelBinarizer().fit(test)
        f1 = f1_score(m.transform(test), m.transform(pred), average='weighted')
        mcc  = matthews_corrcoef(m.transform(test).argmax(axis=1), m.transform(pred).argmax(axis=1))
        zl   = zero_one_loss(m.transform(test), m.transform(pred))*100
        return f1, mcc, zl

    def train_one_epoch(self, epoch):
        self.model.train()
        loss_avg = 0.
        score_count = 0.
        total = 0.
        progress_bar = tqdm(self.train_loader)
        
        # show progress
        for i, data in enumerate(progress_bar):
            progress_bar.set_description('Epoch ' + str(epoch))
            
            images, labels=data
            images = images.to(self.device)
            labels = labels.to(self.device)
            labels_long = labels.to(device=self.device, dtype=torch.int64)
            
            self.model.zero_grad()
            
            pred= self.model(images)
                
            loss = self.loss_function(pred, labels_long)
            score = self._get_accuracy(pred, labels)
            total += labels.size(0)
            score_count += score.sum()
            accuracy = score_count.double() / total
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.)
            self.optimizer.step()
            
            loss_avg +=loss.item()

            progress_bar.set_postfix(
                loss='%.3f' % (loss_avg / (i + 1)),
                score='%.3f' % accuracy)
        return loss_avg / (i + 1), accuracy

    def validate(self, loader):
        self.model.eval()    # Change model to 'eval' mode (BN uses moving mean/var).
    
        running_loss = []
        running_acc = []
        
        with torch.no_grad():

            for i, data in enumerate(loader):
                
                images, labels=data
            
                images = images.to(self.device)
                labels = labels.to(self.device)
                labels_long = labels.to(device=self.device, dtype=torch.int64)
                
                pred = self.model(images)
                loss = self.loss_function(pred, labels_long)
                score = self._get_accuracy(pred, labels).mean()
                
                
            
                running_loss.append(loss.item())
                running_acc.append(score.item())
        self.model.train()
        return np.mean(running_loss), np.mean(running_acc)

    def _weight_init(self, m):    
        if isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        
        elif isinstance(m, nn.BatchNorm1d):
            init.normal_(m.weight.data, mean=1, std=0.02)
            init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.normal_(m.weight.data, mean=1, std=0.02)
            init.constant_(m.bias.data, 0)
        
        elif isinstance(m, nn.Linear):
            init.xavier_normal_(m.weight.data)
            init.normal_(m.bias.data)

    def _get_accuracy(self, y_pred:torch.Tensor, y_true:torch.Tensor, softmax:bool=True):
        "Compute multi class accuracy when `y_pred` and `y_true` are the same size."
        if softmax: y_pred = torch.softmax(y_pred, 1)
        _, y_pred = torch.max(y_pred.data, 1)
        return (y_pred == y_true.data).float()

    def _get_prediction(self, checkpoint, dataloader, num_class):
        # set parameters
        model=checkpoint.load_saved_model(self.model)
        model.eval()
        num_elements = dataloader.len if hasattr(dataloader, 'len') else len(dataloader.dataset)
        batch_size   = dataloader.batchsize if hasattr(dataloader, 'len') else dataloader.batch_size
        num_batches = len(dataloader)

        # initialize labels
        predictions = torch.zeros(num_elements, num_class)
        correct_labels = torch.zeros(num_elements, num_class)
        values = range(num_batches)
        
        with tqdm(total=len(values), file=sys.stdout) as pbar:
        
            with torch.no_grad():
                for i, data in enumerate(dataloader):
                
                    start = i*batch_size
                    end = start + batch_size

                    if i == num_batches - 1:
                        end = num_elements
                    
                    inputs, labels = data
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    
                    out = model(inputs)
                        
                    
                    pred = torch.softmax(out, 1)
                    prob, pred = torch.max(pred.data, 1)
                    
                    predictions[start:end] = pred.unsqueeze(1)
                    correct_labels[start:end] = labels.unsqueeze(1).long()
                
                    pbar.set_description('processed: %d' % (1 + i))
                    pbar.update(1)
                pbar.close()

        predictions = predictions.cpu().numpy()
        correct_labels = correct_labels.cpu().numpy()
        assert(num_elements == len(predictions))
        return predictions, correct_labels
