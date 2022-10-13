import os
import datetime
import shutil
import matplotlib
import matplotlib.pyplot as plt
import json
import torch
from torch.utils import tensorboard
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve,\
    average_precision_score, accuracy_score, confusion_matrix
matplotlib.use('Agg')


class Logger:
    def __init__(self, args, wandb=None, use_hydra=True):
        self.wandb = wandb
        if not use_hydra:
            for past_log in os.listdir(args.log_dir):
                if past_log.split('_', 1)[1] == args.out:
                    ans = input('overwrite "{}" (y/n)'.format(past_log))
                    if ans == 'y' or ans == 'Y':
                        print('move existing directory to dump')
                        past_dir = os.path.join(args.log_dir, past_log)
                        try:
                            shutil.move(past_dir, past_dir.replace(args.log_dir, 'dump'))

                        except:
                            shutil.rmtree(past_dir.replace(args.log_dir, 'dump'))
                            shutil.move(past_dir, past_dir.replace(args.log_dir, 'dump'))

                    else:
                        print('try again')
                        exit()

            else:
                out = datetime.datetime.now().strftime('%y%m%d%H%M') + '_' + args.out

            self.out_dir = os.path.abspath(os.path.join(args.log_dir, out))
        else:
            # hydra directory
            self.out_dir = os.getcwd()

        os.makedirs(os.path.join(self.out_dir, 'models'), exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, 'scores'), exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, 'plot'), exist_ok=True)

        if not use_hydra:
            # save setting
            with open(os.path.join(self.out_dir, 'setting.json'), 'w') as f:
                json.dump(args.__dict__, f, indent=4)

        # tensorboard
        self.writer = tensorboard.SummaryWriter(log_dir=os.path.join(self.out_dir, 'summary'))

    def add_scalar(self, name, scalar, num):
        if self.wandb is not None:
            self.wandb.log({name: scalar}, step=num)
        return self.writer.add_scalar(name, scalar, num)

    def calc_accuracy(self, pred, label, name, num):
        accuracy = np.mean(np.array(label) == np.argmax(pred, axis=1))
        self.add_scalar(name, scalar=accuracy, num=num)
        return accuracy

    def save_model(self, model, model_name):
        torch.save(model.to('cpu').state_dict(), os.path.join(self.out_dir, 'models', model_name))

        return

    def close(self):
        self.writer.close()
        return


