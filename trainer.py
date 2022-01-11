import argparse
import os
import json
from adversarial_training_framework import Adv_Model
import pickle5 as pickle

class Trainer(object):
    #Runs experiments from JSON config file
    def __init__(self):
        config_file = self.get_parser().parse_args().config
        self.unpack_config(config_file)
        self.load_data()
        self.training_parameters = self.set_parameters()
    
    def get_parser(self):
        parser = argparse.ArgumentParser(description='Run adversarial experiments')
        parser.add_argument('config')
        return parser

    def unpack_config(self, config_file):
        pathname = os.path.abspath(os.path.dirname(__file__))
        config = json.load(open(os.path.join(path, config_file), 'r'))
        self.Xtrain_file = os.path.join(pathname, config['Xtrain'])
        self.ytrain_file = os.path.join(pathname, config['ytrain'])
        self.Xvalid_file = os.path.join(pathname, config['Xvalid'])
        self.yvalid_file = os.path.join(pathname, config['yvalid'])
        self.ztrain_file = os.path.join(pathname, config['ztrain'])
        self.zmatch_file = os.path.join(pathname, config['zmatch'])
        self.zvalid_file = os.path.join(pathname, config['zvalid'])
        self.method = config['method']
        self.hyperparameters = config['hyperparameters']
        self.num_classes = config['num_classes']

    def load_data(self):
        with open(self.Xtrain_file, "rb") as filepath:
            self.Xtrain = pickle.load(filepath)
        with open(self.ytrain_file, "rb") as filepath:
            self.ytrain = pickle.load(filepath)
        with open(self.Xvalid_file, "rb") as filepath:
            self.Xvalid = pickle.load(filepath)
        with open(self.yvalid_file, "rb") as filepath:
            self.yvalid = pickle.load(filepath)
        with open(self.ztrain_file, "rb") as filepath:
            self.ztrain = pickle.load(filepath)
        with open(self.zvalid_file, "rb") as filepath:
            self.zvalid = pickle.load(filepath)
        with open(self.zmatch_file, "rb") as filepath:
            self.zmatch = pickle.load(filepath)

    def set_parameters(self):
        parameters = dict()
        parameters['Xtrain'] = self.Xtrain
        parameters['ytrain'] = self.ytrain
        parameters['Xvalid'] = self.Xvalid
        parameters['yvalid'] = self.yvalid
        parameters['ztrain'] = self.ztrain
        parameters['zvalid'] = self.zvalid
        parameters['zmatch'] = self.zmatch
        parameters['method'] = self.method
        parameters['hyperparameters'] = self.hyperparameters
        parameters['num_classes'] = self.num_classes
        return parameters

    def train(self):
        model = Model(self.training_parameters)
        else:
            model.train()
        model.eval()


if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
