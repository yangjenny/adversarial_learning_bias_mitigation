import pandas as pd
import numpy as np
import torch
from torch.autograd import Variable
import itertools
import os
import random
import pickle
import math
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.under_sampling import EditedNearestNeighbours, TomekLinks
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score, roc_curve

# chosen hyperparameters
hyperparameter_list = ['learning_rate', 'num_iters', 'num_nodes', 'num_nodes_adv', 'dropout_rate', 'alpha']

# get_new_control_indices is used when you want to get a specific controls:cases ratio
# Set true when running basic model for the first time
# set false once indices already generated for the first time
get_new_control_indices = False
#Set true if we want to just use data as is (i.e. don't bother with matching controls)
use_data_as_is = False

# metrics used in evaluation
# number labels of the protected variable (k)
def get_metrics(ypred, y, z, hyperparameters, k = 7, yselect = 0, eval_file = None, zpred=None):
    metrics = dict()
    metrics['eval_file'] = eval_file

    # add hyperparameters for experiment
    for i in range(len(hyperparameters)):
        metrics[hyperparameter_list[i]] = hyperparameters[i]

    # performance metrics
    pred = (ypred >= 0.5)
    true_neg, false_neg, false_pos, true_pos = confusion_matrix(pred, y)
    metrics['accuracy'] = accuracy_score(pred, y)
    metrics['recall'] =  true_pos / (false_neg + true_pos)
    print('TN: ' + str(true_neg) +", FN: " + str(false_neg) +", FP: " + str(false_pos)+", TP: " + str(true_pos))
    metrics['precision'] = true_pos / (false_pos + true_pos)
    metrics['specificity'] = true_neg/(true_neg+false_pos)
    #for ppv and npv, set prevalance 
    prev = 0.05
    metrics['ppv'] = (metrics['recall']* (prev))/(metrics['recall'] * prev + (1-metrics['specificity']) * (1-prev))
    metrics['npv'] = (metrics['specificity']* (1-prev))/(metrics['specificity'] * (1-prev) + (1-metrics['recall']) * (prev))
    metrics['f1score'] = 2*(metrics['precision']*metrics['recall'])/(metrics['precision']+metrics['recall'])
    if zpred is not None:
        if k > 2:
            zpred_label = predictmulti(zpred)
            metrics['z-accuracy'] = accuracy_score(zpred_label, z)
        else:
            zpred_label = (zpred >= 0.5)
            metrics['z-accuracy'] = accuracy_score(zpred_label, z)
    metrics['roc_auc'] = roc_auc_score(y, pred)
    return metrics

def predictmulti(prob_list):
    ind_list = []
    for probs in prob_list:
        ind_list.append(np.where(probs==np.max(probs))[0][0])
    return ind_list

def confusion_matrix(ypred, y):
    true_pos = np.sum((ypred == 1) & (y == 1))
    true_neg = np.sum((ypred == 0) & (y == 0))
    false_pos = np.sum((ypred == 1) & (y == 0))
    false_neg = np.sum((ypred == 0) & (y == 1))
    return true_neg, false_neg, false_pos, true_pos

# Model class used
class Adv_Model(object):
    def __init__(self, params):
        self.params = params
        self.method = self.params['method']
        self.adversarial = self.method != 'basic'
        self.num_classes = self.params['num_classes']
        self.hyperparameters = self.params['hyperparameters']
        self.model = self.build_model()
        self.data = self.data_processing()

    def get_indexes(self):
        num_models = []
        for i in range(len(hyperparameter_list)):
            if (i < 3 or i == 4 or self.adversarial):
                num_models.append(range(len(self.hyperparameters[hyperparameter_list[i]])))
            else:
                num_models.append([None]) 
        return itertools.product(*num_models)

    def get_hyperparameters(self, indexes):
        hyperparameters = []
        for i in range(len(indexes)):
            if (i < 3 or i == 4 or self.adversarial):
                hyperparameters.append(self.hyperparameters[hyperparameter_list[i]][indexes[i]])
            else:
                hyperparameters.append(None)
        return hyperparameters

    def params_tostring(self, indexes):
        res = ''
        for i in range(len(hyperparameter_list)):
            if i > 0:
                res += '-'
            if (i < 3 or i == 4 or self.adversarial):
                res += hyperparameter_list[i] + '_' + str(self.hyperparameters[hyperparameter_list[i]][indexes[i]])
        return res

    def create_dir(self, dirname):
        if (not os.path.exists(dirname)):
            os.makedirs(dirname)

    def data_processing(self):
        data = dict()
        i, j = self.params['Xtrain'].shape
        i_valid, j_valid = self.params['Xvalid'].shape
        num_nodes = self.hyperparameters['num_nodes']

        data['Xtrain'] = Variable(torch.tensor(self.params['Xtrain'].values).float())
        data['ytrain'] = Variable(torch.tensor(self.params['ytrain'].values.reshape(i, 1)).float())
        data['Xvalid'] = Variable(torch.tensor(self.params['Xvalid'].values).float())
        data['yvalid'] = Variable(torch.tensor(self.params['yvalid'].values.reshape(i_valid, 1)).float())
        
        if self.num_classes > 2:
            data['ztrain'] = Variable(torch.tensor(self.params['ztrain'].values.reshape(self.params['ztrain'].shape[0],)).long())
            data['zmatch'] = Variable(torch.tensor(self.params['zmatch'].values.reshape(self.params['zmatch'].shape[0],)).long())
            data['zvalid'] = Variable(torch.tensor(self.params['zvalid'].values.reshape(self.params['zvalid'].shape[0],)).long())
        else:
            data['ztrain'] = Variable(torch.tensor(self.params['ztrain'].values.reshape(self.params['ztrain'].shape[0],)).float())
            data['zmatch'] = Variable(torch.tensor(self.params['zmatch'].values.reshape(self.params['zmatch'].shape[0],)).float())
            data['zvalid'] = Variable(torch.tensor(self.params['zvalid'].values.reshape(self.params['zvalid'].shape[0],)).float())

        return data

    def build_model(self):
        models = {}
        for indexes in self.get_indexes():
                models[indexes] = self.build_single_model(indexes)
        return models

    def build_single_model(self, indexes):
        model = dict()

        num_nodes = self.hyperparameters['num_nodes'][indexes[2]]
        i, j = self.params['Xtrain'].shape
        i_valid, j_valid = self.params['Xvalid'].shape
        model['model'] = torch.nn.Sequential(
            torch.nn.Linear(j, num_nodes),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.hyperparameters['dropout_rate'][indexes[4]]),
            torch.nn.Linear(num_nodes, 1),
            torch.nn.Sigmoid(),
        )
        model['loss_function'] = torch.nn.BCELoss(size_average=True)
        model['optimizer'] = torch.optim.Adam(model['model'].parameters(), lr=self.hyperparameters['learning_rate'][indexes[0]])

        if self.adversarial:
            num_nodes_adv = self.hyperparameters['num_nodes_adv'][indexes[3]]
            if self.num_classes > 2:
                num_nodes_out = self.num_classes
            else:
                num_nodes_out = 1

            elif self.method == 'adv':
                n_adv = 2
            if (self.num_classes > 2):
                model['adversarial_model'] = torch.nn.Sequential(
                    torch.nn.Linear(n_adv, num_nodes_adv),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(self.hyperparameters['dropout_rate'][indexes[4]]),
                    torch.nn.Linear(num_nodes_adv, num_nodes_out),
                    torch.nn.Softmax(),
                )
            else:
                model['adversarial_model'] = torch.nn.Sequential(
                    torch.nn.Linear(n_adv, num_nodes_adv),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(self.hyperparameters['dropout_rate'][indexes[4]]),
                    torch.nn.Linear(num_nodes_adv, num_nodes_out),
                    torch.nn.Sigmoid(),
                )
            if (self.num_classes > 2):
                model['adversarial_loss_function'] = torch.nn.CrossEntropyLoss(size_average=True)
            else:
                model['adversarial_loss_function'] = torch.nn.BCELoss(size_average=True)
            model['adversarial_optimizer'] = torch.optim.Adam(model['adversarial_model'].parameters(), lr=self.hyperparameters['learning_rate'][indexes[0]])

        return model

    def train(self):
        for indexes in self.get_indexes():
            self.train_single_model(indexes)

    def load_trained_models(self):
        for indexes in self.get_indexes():
            modelfile = 'model/model-basic.pth'
            self.model[indexes]['model'] = torch.load(modelfile)
            advmodelfile = 'adv/model-adv.pth'
            self.model[indexes]['adversarial_model'] = torch.load(advmodelfile)


    def train_single_model(self, indexes):
        # Load in model and data
        model = self.model[indexes]['model']
        loss_function = self.model[indexes]['loss_function']
        optimizer = self.model[indexes]['optimizer']
        Xtrain = self.data['Xtrain']
        print('original xtrain: ', Xtrain.shape)
        Xvalid = self.data['Xvalid']
        ytrain = self.data['ytrain']
        yvalid = self.data['yvalid']
        ztrain = self.data['ztrain']
        zmatch = self.data['zmatch']
        zvalid = self.data['zvalid']
        
        if use_data_as_is == False:
            matched_cohort_indices = []
            match_number = 20
            idx_control = [i for i in range(len(ytrain)) if ytrain[i] == 0]
            control_data = Xtrain[idx_control,:]
            control_y = [ytrain[i] for i in idx_control]
            control_z = [ztrain[i] for i in idx_control]
            control_age = [zmatch[i] for i in idx_control]
            idx_case = [i for i in range(len(ytrain)) if ytrain[i] == 1]
            case_data = Xtrain[idx_case,:]
            case_y = [ytrain[i] for i in idx_case]
            case_z = [ztrain[i] for i in idx_case]
            case_age = [zmatch[i] for i in idx_case]

            if get_new_control_indices == True:
                count = 1
                for index in idx_case:
                    print(str(count))
                    patient_data = Xtrain[index,:]
                    patient_age = zmatch[index].numpy()
                    patient_z = ztrain[index].numpy()
                    age_condition = control_age == patient_age
                    z_condition = control_z == patient_z
                    matched_indices_bool = age_condition & z_condition
                    matched_indices= np.array(idx_control)[matched_indices_bool]
                    random.seed(0)
                    random.shuffle(matched_indices)
                    valid_indices = list(set(matched_indices)-set(matched_cohort_indices))[:match_number]
                    matched_cohort_indices.extend(valid_indices)
                    count=count+1
                with open(os.path.join('control_indices_%i.pkl' % (match_number)),'wb') as f:
                            pickle.dump(matched_cohort_indices,f)
            else:
                with open(os.path.join('control_indices_%i_ethnicity.pkl' % (match_number)),'rb') as f:
                    matched_cohort_indices = pickle.load(f)
            control_matched_data = Xtrain[matched_cohort_indices,:]
            print('new xtrain matched control: ', control_matched_data.shape)
            control_matched_z = [ztrain[i] for i in matched_cohort_indices]
            control_matched_y = [ytrain[i] for i in matched_cohort_indices]
            Xtrain = np.concatenate((control_matched_data, case_data), axis=0)
            ytrain = np.concatenate((control_matched_y + case_y),axis=None)
            ztrain = np.concatenate((control_matched_z + case_z),axis=None)
        
        if self.adversarial:
            resample = SMOTEENN(enn=EditedNearestNeighbours(sampling_strategy='majority'), random_state=25)
            #resample = SMOTETomek(tomek=TomekLinks(sampling_strategy='majority'), random_state=25)
            Xztrain = torch.from_numpy(np.append(Xtrain, ztrain.reshape(len(ztrain), 1), axis=1))
            Xztrain, ytrain = resample.fit_resample(Xztrain, ytrain)
            Xtrain = torch.from_numpy(Xztrain[:,:-1])
            ztrain = torch.from_numpy(Xztrain[:,-1])
            ytrain = torch.from_numpy(ytrain.reshape(len(ytrain), 1))
        else:
            #resample = SMOTETomek(tomek=TomekLinks(sampling_strategy='majority'), random_state=25)
            resample = SMOTEENN(enn=EditedNearestNeighbours(sampling_strategy='majority'), random_state=25)
            Xtrain, ytrain = resample.fit_resample(Xtrain, ytrain)
            Xtrain = torch.from_numpy(Xtrain)
            ytrain = torch.from_numpy(ytrain.reshape(len(ytrain), 1))
        
        if self.adversarial:
            adversarial_model = self.model[indexes]['adversarial_model']
            adversarial_loss_function = self.model[indexes]['adversarial_loss_function']
            adversarial_optimizer = self.model[indexes]['adversarial_optimizer']

        model.train()

        # Save models and metrics
        self.create_dir('metrics/')
        self.create_dir('model/')
        if self.adversarial:
            self.create_dir('adv/')
        hyperparam_values = self.params_tostring(indexes)
        metrics_file = 'metrics/metrics.csv'
        metrics = []
        modelfile = 'model/model.pth'
        if self.adversarial:
            advfile = 'adv/model-adv.pth'
        
        #save training loss
        train_loss_list = []
        train_adversarial_loss_list = []
        train_combined_loss_list = []
        valid_loss_list = []
        valid_adversarial_loss_list = []
        valid_combined_loss_list = []
        epoch_list = []

        for t in range(self.hyperparameters['num_iters'][indexes[1]]):
            # Forward step
            ypred_train = model(Xtrain.float())
            loss_train = loss_function(ypred_train, ytrain)

            ypred_valid = model(Xvalid)
            loss_valid = loss_function(ypred_valid, yvalid)

            if self.adversarial:
                elif self.method == 'adv':
                    adversarial_input_train = torch.cat((ypred_train, ytrain), 1)
                    adversarial_input_valid = torch.cat((ypred_valid, yvalid), 1)

                zpred_train = adversarial_model(adversarial_input_train)
                #add .long() if doing multiclass Z
                adversarial_loss_train = adversarial_loss_function(zpred_train.squeeze(), ztrain.long())
                zpred_valid = adversarial_model(adversarial_input_valid)
                adversarial_loss_valid = adversarial_loss_function(zpred_valid.squeeze(), zvalid.long())
                
                #ORIGINAL LOSS FUNCTION
                #combined_loss_train = loss_train - self.hyperparameters['alpha'][indexes[5]] * adversarial_loss_train
                #CORRECTION TERM ADDED
                combined_loss_train = loss_train - self.hyperparameters['alpha'][indexes[5]] * adversarial_loss_train + loss_train/adversarial_loss_train
                #PROJECTION TERM and INCREASING ALPHA WITH ITERATION ADDED
                #combined_loss_train = loss_train - math.sqrt(t) * adversarial_loss_train + loss_train/adversarial_loss_train
                #ORIGINAL LOSS FUNCTION
                #combined_loss_valid = loss_valid - self.hyperparameters['alpha'][indexes[5]] * adversarial_loss_valid
                ######PROJECTION TERM ADDED
                combined_loss_valid = loss_valid - self.hyperparameters['alpha'][indexes[5]] * adversarial_loss_valid + loss_valid/adversarial_loss_valid
                ######PROJECTION TERM and INCREASING ALPHA WITH ITERATION ADDED########
                #combined_loss_valid = loss_valid - math.sqrt(t) * adversarial_loss_valid + loss_valid/adversarial_loss_valid
            
            
            # Training log
            if t % 100 == 0:
                print('Iteration: {}'.format(t))
                epoch_list.append(t)
                if self.adversarial:
                    print('Predictor train loss: {:.4f}'.format(loss_train))
                    train_loss_list.append(loss_train.item())
                    print('Predictor valid loss: {:.4f}'.format(loss_valid))
                    valid_loss_list.append(loss_valid.item())
                    print('Adversary train loss: {:.4f}'.format(adversarial_loss_train))
                    train_adversarial_loss_list.append(adversarial_loss_train.item())
                    print('Adversary valid loss: {:.4f}'.format(adversarial_loss_valid))
                    valid_adversarial_loss_list.append(adversarial_loss_valid.item())
                    print('Combined train loss:  {:.4f}'.format(combined_loss_train))
                    train_combined_loss_list.append(combined_loss_train.item())
                    print('Combined valid loss:  {:.4f}'.format(combined_loss_valid))
                    valid_combined_loss_list.append(combined_loss_valid.item())
                else:
                    print('Train loss: {:.4f}'.format(loss_train))
                    train_loss_list.append(loss_train.item())
                    print('Valid loss: {:.4f}'.format(loss_valid))
                    valid_loss_list.append(loss_valid.item())
            
            # Save model
            if t > 0 and t % 10000 == 0:
                torch.save(model, modelfile)
                if self.adversarial:
                    torch.save(adversarial_model, advfile)

            # Backward step
            if self.adversarial:
                # adv update
                adversarial_optimizer.zero_grad()
                adversarial_loss_train.backward(retain_graph=True)
                
                # pred update
                optimizer.zero_grad()
                combined_loss_train.backward()
                adversarial_optimizer.step()
            else:
                optimizer.zero_grad()
                loss_train.backward()

            optimizer.step()
        
        loss_data = {'epoch': epoch_list, 'train_loss': train_loss_list,'valid_loss': valid_loss_list, 'train_adversarial_loss': train_adversarial_loss_list, 'valid_adversarial_loss': valid_adversarial_loss_list, 'train_combined_loss': train_combined_loss_list, 'valid_combined_loss': valid_combined_loss_list}
        loss_df = pd.DataFrame.from_dict(loss_data, orient='index')
        loss_df = loss_df.transpose()
        loss_df.to_csv("loss_metrics.csv")
        
        if self.adversarial:
            plt.plot(epoch_list, train_loss_list, color='blue')
            plt.plot(epoch_list, valid_loss_list, color='red')
            plt.plot(epoch_list, train_adversarial_loss_list, color='blue', linestyle='--')
            plt.plot(epoch_list, valid_adversarial_loss_list, color='red', linestyle='--')
            plt.show()
        else:
            plt.plot(epoch_list, train_loss_list, color='blue')
            plt.plot(epoch_list, valid_loss_list, color='red')
            plt.show()
            
        # save final model
        torch.save(model, modelfile)
        if self.adversarial:
            torch.save(adversarial_model, advfile)
        writer.close()

    def evaluate(self):
        evalfile = 'metrics.csv'
        valid_metrics = []
        for indexes in self.get_indexes():
            valid_metrics.append(self.evaluate_single_model(indexes))
        pd.concat(valid_metrics).to_csv(evalfile)

    def evaluate_single_model(self, indexes):
        model = self.model[indexes]['model']
        Xtrain = self.data['Xtrain']
        Xvalid = self.data['Xvalid']
        ytrain = self.data['ytrain']
        yvalid = self.data['yvalid']
        ztrain = self.data['ztrain']
        zvalid = self.data['zvalid']

        model.evaluate()

        ypred_valid = model(Xvalid)
        zpred_valid = None
        if self.adversarial:
            adversarial_model = self.model[indexes]['adversarial_model']
            adversarial_model.evaluate()
            elif self.method == 'adv':
                adversarial_input_valid = torch.cat((ypred_valid, yvalid), 1)
                zpred_valid = adversarial_model(adversarial_input_valid)

        if zpred_valid is not None:
            metrics_valid = pd.DataFrame(get_metrics(ypred_valid.data.numpy(), yvalid.data.numpy(), zvalid.data.numpy(), self.get_hyperparameters(indexes), k=self.num_classes, eval_file='valid_set', zpred=zpred_valid.data.numpy()), index=[0])
        else:
            metrics_valid = pd.DataFrame(get_metrics(ypred_valid.data.numpy(), yvalid.data.numpy(), zvalid.data.numpy(), self.get_hyperparameters(indexes), k=self.num_classes, eval_file='valid_set'), index=[0])
        return metrics_valid
