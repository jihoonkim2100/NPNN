# -*- coding: utf-8 -*-
################################################################################
"""
This is a Neural Network Programming Project (NNPP) module for Neurocognitive
Methods and Programming (SoSe 2020). This module provides classes for
preprocessing and multi-, and binary classification using simple neural network
based on the "Haxby dataset" <http://data.pymvpa.org/datasets/haxby2001/> with
.csv labels. Those are required to run this module:
    - python : 3.6.10
    - nibabel : 3.1.1
    - nipype : 1.6.0 dev0
    - nilearn : 0.6.2
    - pandas : 1.0.5
    - matplotlib : 3.2.2
    - tensorflow : 1.13.1
    - keras : 2.2.4
    - jupyter notebook : 6.0.3
And "You must install FSL to work with the nipype in this module".

This was tested on the docker "jihoonkim2100/nnpp"
<https://hub.docker.com/r/jihoonkim2100/nnpp> environment
which leverage tensorflow and keras based on "nipype/nipype":
    - docker version : 19.03.1
    - docker image OS : Debian GNU/Linux 9
    - host OS : Windows 10 Home

Belows are the reference for this project:
    - scikit-learn https://scikit-learn.org/stable/
    - nilearn https://nilearn.github.io/index.html
    - nipype https://nipype.readthedocs.io/en/latest/index.html
    - keras https://keras.io/

Author : JiHoon Kim
Last-modified : 13th July 2020
"""
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

import nibabel as nib
import nilearn.image as nimg
from nilearn.input_data import NiftiMasker
from nilearn.plotting import plot_anat
from nipype.interfaces import fsl
from nipype.interfaces.fsl import BET
from sklearn.model_selection import GridSearchCV

import keras
from keras import models
from keras import layers
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam 
from keras.utils import np_utils
from keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
print ( "--sys.version—") 
print (sys . version)

# for reproducibility
np.random.seed(17072020)


class DataAcquisiton:
    """
    For data acquistion,
    * newfolder : Create new folder.
    * inputd : Load the input data.
    * outputd : Save the output data.
    * reorder : Coordinate RAS to LAS.

    Author : JiHoon Kim
    Last-modified : 13th July 2020
    """
    def setdata(self,folder,in_path,data,data2,data3,out_path):
        """
        setdata requires a total of seven paramters :
        self, folder, in_path, data, data2, data3, and out_path.
        """
        self.folder = folder
        self.in_path = in_path
        self.data = data
        self.data2 = data2
        self.data3 = data3
        self.out_path = out_path
        
    def newfolder(self,folder):
        """
        This function for creating a new folder.
        folder : read the string, new folder's name.

        So that it makes a folder.
        """ 
        os.mkdir(folder)
    
    def inputd(self,in_path):
        """
        This function for loading the data.
        in_path : read the string, directory address of data.
        
        Returns a nibabel.nifti1.Nifti1Image.
        """
        print(in_path,"loaded")
        img = nib.load(in_path)
        return img
       
    def outputd(self,data,out_path):
        """
        This function for saving the data.
        data : a nibabel.nifti1.Nifti1Image, data to be saved.
        out_path : string, directory address to save.
        """
        nib.save(data,out_path)
        print(out_path,"saved")
        
    def reorder(self,data):
        """
        This function reorder LAS to RAS.
        data : nibabel.nifti1.Nifti1Image.
        
        Returns a RAS coordinated nibabel.nifti1.Nifti1Image.
        """
        # Check the coordination and reorder the data LAS to RAS
        coord = nib.aff2axcodes(data.affine)
        if 'L' == coord[0]:
            data = nimg.reorder_img(data,resample=None)
            a = f"Reorder {nib.aff2axcodes(data.affine)}"
            print(a,"completed")
        else:
            a = f"Original {coord}"
            print(a,"checked")
        return data
  
    def acquisition(self,folder,in_path,out_path):
        """
        This function load, reorder, and save the data.
        It is dependent on the class name: func.
        folder: read the string, new folder's name.
        in_path: read the list, element: string, directory address of data.
        out_path: read the string, name of the new data.

        returns a list, element: string, directory address of new data.
        """
        print('#################_Data_acquisition_started_#################')
        acquisition = time.time()

        # Check and if not make the directory
        if not os.path.isdir(folder):
            func.newfolder(folder)

        # Load, reorder, and save the data
        output_list = []
        for i in range(len(in_path)):
            output = folder+'/'+out_path[i]
            output_list.append(output)
            func.outputd(func.reorder(func.inputd(in_path[i])),output)
        print("acquisition_time :", "%.2fs" %(time.time() - acquisition))
        print("#################_Data_acquisition_completed_###############")
        return output_list

class DataPreprocessing(DataAcquisiton):
    """
    For Preprocessing of fMRI Data,
    * mask_image : Mask image template.
    * mcflirt : Motion correction.
    * niftimasker : Smooth, normalize.

    Author : JiHoon Kim
    Last-modified : 13th July 2020
    """
    def mask_image(self,folder,in_path,out_path):
        """
        This function for creating mask_image.
        It is dependent on the class name: func.
        folder : read the string, new folder's name.
        in_path : read the list, element: string, directory address of data.
        out_path : read the string, name of the new data.

        For complete details, see the BET() documentation.
        <https://nipype.readthedocs.io/en/latest/interfaces.html>
        """
        print('#################_Mask_image_started_#######################')
        preprocessing = time.time()

        # Check and if not make the directory
        if not os.path.isdir(folder):
            func.newfolder(folder)

        # Create, and save the data : anatomy and whole brain mask image 
        for i in range(len(in_path)):
            output = folder+'/'+out_path[i]
            print(in_path[i], "mask image started")
            skullstrip = BET(in_file=in_path[i],
                             out_file=output,
                             mask=True)
            skullstrip.run()
            print(output,"mask image completed")
        print("computation_time :","%.2fs" %(time.time() - preprocessing))
        print('#################_Mask_image_completed_#####################')

    def mcflirt(self,folder,in_path):
        """
        This function for motion correction.
        It is dependent on the class name: func.
        folder : read the string, new folder's name.
        in_path : read the list, element: string, directory address of data.

        returns a list, element: string, directory address of new data.
        For complete details, see the fsl.MCFLIRT() documentation.
        <https://nipype.readthedocs.io/en/latest/interfaces.html>
        """ 
        print('#################_Data_motion_correction_started_###########')
        preprocessing = time.time()

        # Check and if not make the directory
        if not os.path.isdir(folder):
            func.newfolder(folder)

        # Proceed the motion correction and save the data
        output_list = []        
        for i in range(len(in_path)):
            print(in_path[i],"motion correction started")
            file = in_path[i].split('/')
            output = folder+'/'+file[1]
            output_list.append(output)
            
            mcflt = fsl.MCFLIRT()
            mcflt.inputs.in_file = in_path[i]
            mcflt.inputs.cost = 'mutualinfo'
            mcflt.inputs.out_file = output
            mcflt.cmdline
            mcflt.run()
            print(output,"motion correction completed")
        print("computation_time :","%.2fs" %(time.time() - preprocessing))
        print('#################_Data_motion_correction_completed_#########')
        return output_list
    
    def niftimasker(self,in_path,out_path):
        """
        This function for masking whole brain image.
        in_path : read the list, element: string, directory address of data.
        out_path : read the list, element: string, directory address of data.
        For complete details, see the nilearn.input_data.NiftiMasker.
        <https://nilearn.github.io/modules/reference.html>
        """
        print('#################_Data_preprocessing_started_###############')
        preprocessing = time.time()

        # Smoothing fwhm : 12, mask, standardize, detrend, and resampling.
        for i in range(len(in_path)):
            print(in_path[i])
            print("smoothing_fwhm : 12, and normalization started")
            masker = NiftiMasker(mask_img=out_path[i],smoothing_fwhm= 12,
                                 standardize=True,detrend=True,
                                 mask_strategy='template')
            func_data = masker.fit_transform(in_path[i])
            print(in_path[i],'to func_data')
            print('smoothing_fwhm : 12, and normalization completed')
        print("computation_time :", "%.2fs" %(time.time() - preprocessing))
        print('#################_Data_preprocessing_completed_#############')
        return func_data

class NN(DataPreprocessing):
    """
    For Nueral Network Analysis,
    * categorical_label : Load the categorical label.
    * categorical_dataset : Prepare the dataset, train and test set.
    * grid_search : Proceed the gridsearch on the train set.
    * cross_validation : Do the cross validation on the train set.
    * test_model : Train, test, and save the model on the dataset.

    Author : JiHoon Kim
    Last-modified : 13th July 2020 
    """
    def categorical_label(self,in_path):
        """
        This function load the multiclass label of the dataset.
        in_path : read the string, directory address of the label data.
        
        return two lists (numbers_list, labels_list which contain labels and
        sample order number of fMRI data), elements are integer.
        """
        print('#################_Categorical_label_checked_################')
        preprocessing = time.time()

        # Load the label, .csv file
        labels = pd.read_csv(in_path)
        stimuli = labels['label']

        # Select the stimuli labels,
        # 0 : 'scissors', 1 :'face', 2 : 'cat', 3 : 'shoe', 4 : 'house',
        # 5 : 'scrambledpix', 6 : 'bottle', and 7 : 'chair'.
        numbers_list = []
        labels_list = []
        num = 0
        for i in stimuli: 
            num += 1
            if i == 'scissors':
                numbers_list.append(num)
                labels_list.append(0)
            elif i == 'face':
                numbers_list.append(num)
                labels_list.append(1)
            elif i == 'cat':  
                numbers_list.append(num)
                labels_list.append(2)
            elif i == 'shoe':
                numbers_list.append(num)
                labels_list.append(3)
            elif i == 'house':
                numbers_list.append(num)
                labels_list.append(4)
            elif i == 'scrambledpix':
                numbers_list.append(num)
                labels_list.append(5)
            elif i == 'bottle':
                numbers_list.append(num)
                labels_list.append(6)
            elif i == 'chair':
                numbers_list.append(num)
                labels_list.append(7)
        print('labels_list:',len(labels_list),'loaded')
        print("computation_time :","%.2fs" %(time.time() - preprocessing))
        print('#################_Categorical_label_loaded_#################')
        return numbers_list, labels_list

    def categorical_dataset(self,in_path,data,data2):
        """
        This function prepare the dataset,
        in_path : read a list which contains labels data.
        data : read a np.array which contains feature data.
        data2 : read a list which contains fMRI orders data.

        return four np.array, and one integer,
        train_data, test_data, train_labels, test_labels, and input size. 
        """
        print('#################_Categorical_feature_checked_##############')
        preprocessing = time.time()

        # Collect only labels stimuli fMRI data
        func_data = []
        labels_list = in_path
        for i in data2:
            func_data.append(data[i])
        
        # Create train set and test set with 5:1 ratio
        fold = int(len(func_data)*5/6)
        train_data = np.array(func_data[:fold])
        test_data = np.array(func_data[fold:])

        train_labels = np.array(labels_list[:fold])
        test_labels = np.array(labels_list[fold:])
        print("train_data, test_data :",len(train_data),len(test_data))

        # Normalize the dataset
        mean = train_data.mean(axis=0)
        train_data -= mean
        std = train_data.std(axis=0)
        train_data /= std

        test_data -= mean
        test_data /= std
        
        train_labels = to_categorical(train_labels)
        test_labels = to_categorical(test_labels)    
        
        size = len(train_data[1])
        print("Dataset standardization completed")
        print("size:",len(train_data[1])," train_data:",len(train_data),
              "and","test_data:",len(test_data),"loaded")
        print("computation_time :","%.2fs" %(time.time() - preprocessing))
        print('#################_Categorical_feature_loaded_###############')
        return train_data,test_data,train_labels,test_labels,size 
    
    def grid_search(self,data,data2):
        """
        This function for grid search,
        data : read a np.array which contains train_data.
        data2 : read a list which contains train_labels.
        """
        print('#################_Grid_search_started_######################')
        preprocessing = time.time()

        def gridsearch_model(dropout_rate,neurons,neurons2):
            """
            This function for building a gridsearch_model,
            dropout_rate : float, 0 to 1 normally 0.1, 0.3, 0.5 etc.
            neurons : integer, first dropout layer parameter.
            neurons2 : integer, second dropout layer parameter.

            returns a model.
            """  
            model = models.Sequential()
            model.add(Dense(neurons,
                            input_shape = (size, ),
                            activation = 'relu'))
            model.add(Dropout(dropout_rate))
            model.add(Dense(neurons2,
                            activation = 'relu'))
            model.add(Dropout(dropout_rate))
            model.add(Dense(8,
                            activation = 'softmax'))   
    
            model.compile(optimizer = 'Adam',
                          loss = 'categorical_crossentropy',
                          metrics = ['acc'])
            return model

        model = KerasClassifier(build_fn = gridsearch_model,
                                batch_size = 50, epochs = 100)

        # Gridsearch on hyper parameters
        dropout_rate = [0.1, 0.3, 0.5]
        neurons = [20, 25]
        neurons2 = [10, 15]
        epochs = [50, 100]
        batch_size = [25, 50]

        param_grid = dict(dropout_rate = dropout_rate,
                          epochs = epochs, batch_size = batch_size,
                          neurons = neurons,neurons2 = neurons2)
        grid = GridSearchCV(estimator = model, param_grid = param_grid,
                            n_jobs=-1)
        grid_result = grid.fit(data,data2)

        # Summarize results
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean,stdev,param in zip(means,stds,params):
            print("%f (%f) with: %r" % (mean,stdev,param))
        print("Best: %f using %s" % (grid_result.best_score_,
                                     grid_result.best_params_))
        print("computation_time :","%.2fs" %(time.time() - preprocessing))
        print('#################_Grid_search_completed_####################')              
       
    def cross_validation(self,in_path,data,data2,data3):
        """
        This function for cross validation,
        in_path : read a list, which contains 0 : integer 3, 4, 5 (K-fold),
        1 : integer 25, 32 (batch_size), 2 : float 0.1 to 1.0 (drop_out_rate),
        3 : integer 50, 100 (num_epochs), 4 : integer 10, 20 (neuron), and
        5 : integer 20, 30 (neurons2).
        data : 2d np array, train_data.
        data2 : 2d np array, train_labels.
        """
        print('#################_Cross_validation_checked_#################')
        preprocessing = time.time()

        def cross_validation_model(drop_out,neuron,neuron2):
            """
            This function for building a cross_validation_model,
            dropout_rate : float, 0 to 1 normally 0.1, 0.3, 0.5 etc.
            neurons : integer, first dropout layer parameter.
            neurons2 : integer, second dropout layer parameter.

            returns a model
            """
            model = models.Sequential()
            model.add(layers.Dense(neuron,activation = 'relu',
                                   input_shape = (size, )))
            model.add(layers.Dropout(drop_out))
            model.add(layers.Dense(neuron2,activation = 'relu'))
            model.add(layers.Dropout(drop_out))
            model.add(layers.Dense(8,activation = 'softmax'))
            model.compile(optimizer = 'Adam',
                          loss = 'categorical_crossentropy',
                          metrics = ['acc'])
            return model 
        
        train_data = data2
        train_labels = data3
        
        k = 5
        num_val_samples = len(train_data)//k
        num_epochs = 200
        all_acc_histories = []
        all_loss_histories = []
        all_val_loss_histories = []
        all_val_acc_histories = []

        # Cross validation check, train set and validation set with 4:1 ratio
        for i in range(k):
            print('current fold #', i+1)
            val_data = train_data[i*num_val_samples: (i+1)*num_val_samples]
            val_labels = train_labels[i*num_val_samples: (i+1)*num_val_samples]
    
            partial_train_data = np.concatenate(
                [train_data[:i*num_val_samples],
                 train_data[(i+1)*num_val_samples:]],
                axis=0)
            partial_train_labels = np.concatenate(
                [train_labels[:i*num_val_samples],
                 train_labels[(i+1)*num_val_samples:]],
                axis=0)
    
            model = cross_validation_model(data[0],data[1],data[2])
            history = model.fit(partial_train_data,partial_train_labels,
                                validation_data=(val_data,val_labels),
                                epochs = num_epochs,batch_size = in_path,
                                verbose = 0)

            # Shows the acc and loss plot
            fig, loss_ax = plt.subplots()
            acc_ax = loss_ax.twinx()

            plt.title('Training and validation acc and loss')
            loss_ax.plot(history.history['loss'],'y',label = 'train loss')
            loss_ax.plot(history.history['val_loss'],'r',label = 'val loss')

            acc_ax.plot(history.history['acc'],'b',label = 'train acc')
            acc_ax.plot(history.history['val_acc'],'g',label = 'val acc')

            loss_ax.set_xlabel('epoch')
            loss_ax.set_ylabel('loss')
            acc_ax.set_ylabel('accuracy')

            loss_ax.legend(loc = 'upper left')
            acc_ax.legend(loc = 'lower left')
            plt.show()

            # Save the histories: acc, loss, val_acc, and val_loss
            all_acc_histories.append(history.history['acc'])
            all_loss_histories.append(history.history['loss'])
            all_val_acc_histories.append(history.history['val_acc'])
            all_val_loss_histories.append(history.history['val_loss'])

        # Compute the average histories
        average_acc_history = [
            np.mean([x[i] for x in all_acc_histories])
            for i in range(num_epochs)]
        average_loss_history = [
            np.mean([x[i] for x in all_loss_histories]) 
            for i in range(num_epochs)]
        average_val_acc_history = [
            np.mean([x[i] for x in all_val_acc_histories]) 
            for i in range(num_epochs)]
        average_val_loss_history = [
            np.mean([x[i] for x in all_val_loss_histories]) 
            for i in range(num_epochs)]

        # Shows the average acc and loss plot
        fig, loss_ax = plt.subplots()
        acc_ax = loss_ax.twinx()
        plt.title('Average training and validation acc and loss')
        loss_ax.plot(average_loss_history,'y',label = 'train loss')
        loss_ax.plot(average_val_loss_history,'r',label = 'val loss')
        acc_ax.plot(average_acc_history,'b',label = 'train acc')
        acc_ax.plot(average_val_acc_history,'g',label = 'val acc')
        loss_ax.set_xlabel('epoch')
        loss_ax.set_ylabel('loss')
        acc_ax.set_ylabel('accuracy')
        loss_ax.legend(loc = 'upper left')
        acc_ax.legend(loc = 'lower left')
        plt.show()

        # Shows the average loss plot
        fig, loss_ax = plt.subplots()
        plt.title('Average training and validation loss')
        loss_ax.plot(average_loss_history,'y',label = 'train loss')
        loss_ax.plot(average_val_loss_history,'r',label = 'val loss')
        loss_ax.set_xlabel('epoch')
        loss_ax.set_ylabel('loss')
        loss_ax.legend(loc = 'upper left')
        plt.show()

        # Shows the average acc plot
        fig, acc_ax = plt.subplots()
        plt.title('Average training and validation acc')
        acc_ax.plot(average_acc_history,'b',label = 'train acc')
        acc_ax.plot(average_val_acc_history,'g',label = 'val acc')
        loss_ax.set_xlabel('epoch')
        acc_ax.set_ylabel('accuracy')
        acc_ax.legend(loc = 'lower right')
        plt.show()
        print("computation_time :","%.2fs" %(time.time() - preprocessing))
        print('#################_Cross_validation_completed_###############')
        
    def test_model(self,data,data2):
        """
        This function for train, test, and save the model,
        data : read a integer, num_epochs.
        data2 : read a list which 0 : integer (batch_size),
        1 : float (drop_out_rate), 2 : integer (neuron), 3: integer (neuron2).
        """    
        print('#################_Test_model_started_#######################')
        preprocessing = time.time()
        
        def build_model(drop_out,neuron,neuron2):
            """
            This function for building a model,
            dropout_rate : float, 0 to 1 normally 0.1, 0.3, 0.5 etc.
            neurons : integer, first dropout layer parameter.
            neurons2 : integer, second dropout layer parameter.

            returns a model.
            """ 
            model = models.Sequential()
            model.add(layers.Dense(neuron,activation = 'relu',
                                   input_shape = (size, )))
            model.add(layers.Dropout(drop_out))
            model.add(layers.Dense(neuron2,activation = 'relu'))
            model.add(layers.Dropout(drop_out))
            model.add(layers.Dense(8,activation = 'softmax'))
            model.compile(optimizer = 'Adam',loss = 'categorical_crossentropy',
                          metrics = ['acc'])
            return model

        model = build_model(data2[0],data2[1],data2[2])
        print(f"-- Multiclassification_model, epoch: {data} --")
        model.summary()

        # Train a model
        model.fit(train_data,train_labels,epochs=data,
                  batch_size = int(batch_size), verbose = 0)
        print("-- Evaluate --")

        # Test a model
        score = model.evaluate(test_data,test_labels,verbose = 1)
        print('loss :',score[0])
        print("%s : %.2f%%" %(model.metrics_names[1],score[1]*100))

        # Save a model
        model.save(f"model_epoch:{data}_weight.h5")
        with open(f"model_epoch:{data}_architecture.json",'w') as f:
            f.write(model.to_json())
        print(f"model_epoch : {data}, weight_and_architecture_saved")
        print("computation_time :","%.2fs" %(time.time() - preprocessing))
        print('#################_Test_model_completed_#####################')

class Extra(NN):
    """
    Extra module
    * anat_mask : Show the anatomy image and whole brain mask image.
    * func_img : Display the fMRI signal, original and motion correction one.
    * binary_classification : Run face vs. house binary classification.

    Author : JiHoon Kim
    Last-modified : 13th July 2020
    """
    def anat_mask(self,in_path,out_path):
        """
        This function for showing anat_mask.
        in_path : read the list, element: string, directory address of image.
        out_path : read the list, element: string, for title of the image.
        """
        for i in range(len(in_path)):
            plot_anat(in_path[i],title = out_path[i],
                      display_mode = 'ortho',dim = -1,
                      draw_cross = False, annotate = False)
            
    def func_img(self,in_path,data,data2,out_path):
        """
        This function for displaying func_img.
        in_path : read the string, directory address of original_image.
        data : read the string, for title of original_image.
        data2 : read the string, for title of other_image.
        out_path : read the string, directory address of other_image.
        """
        # Load the images
        rod = func.inputd(in_path)
        prp = func.inputd(out_path)
                         
        # Plot a representative voxel
        x,y,z = 31,35,42
        plt.figure(figsize=(12,4))
        plt.plot(rod.get_data()[x,y,z,:])
        plt.plot(prp.get_data()[x,y,z,:])
        plt.legend([data,data2])
        
    def binary_classification(self,data,data2,data3):
        """
        This function for binary classification.
        data : read the 2d np array, fMRI dataset.
        data2 : read the 2d np array, labels list.
        data3 : read the 2d np array, fMRI order number list.
        """
        print('#################_Binary_classification_started_############')
        preprocessing = time.time()

        func_data = data
        labels_list =data2
        numbers_list = data3
        binary_label_list = []
        binary_data = []

        # Collect only face and house labels stimuli and fMRI data
        for i in range(len(labels_list)):
            if labels_list[i] == 1:
                binary_data.append(func_data[numbers_list[i]])
                binary_label_list.append(0) # face stimuli
            elif labels_list[i] == 4:
                binary_data.append(func_data[numbers_list[i]])
                binary_label_list.append(1) # house stimuli

        # Create train set and test set with 5:1 ratio
        fold = int(len(binary_data)*5/6)
        train_data = np.array(binary_data[:fold])
        test_data = np.array(binary_data[fold:])
        train_labels = binary_label_list[:fold]
        test_labels = binary_label_list[fold:]

        # Normalize the dataset
        mean = train_data.mean(axis=0)
        train_data -= mean
        std = train_data.std(axis=0)
        train_data /= std

        test_data -= mean
        test_data /= std

        # Create train set and validation set with 4:1 ratio
        fold2 = int(len(train_data)*1/5)
        binary_partial_train_data = train_data[fold2:]
        binary_val_data = train_data[:fold2]
        binary_partial_train_labels = train_labels[fold2:]
        binary_val_labels = train_labels[:fold2]
        print("train_data:",len(train_data)," test_data :",len(test_data))
        print("partial_train_data:",len(binary_partial_train_data),
              " val_data:",len(binary_val_data))
        print("Dataset standardization completed")
        print("size:",len(train_data[1])," train_data:",len(train_data),
              "and","test_data:",len(test_data),"loaded")
    
        # Build the binary classification model
        model = models.Sequential()
        model.add(layers.Dense(25,activation = 'relu',
                               input_shape = (len(train_data[1]), )))
        model.add(layers.Dropout(0.3))
        model.add(layers.Dense(15,activation = 'relu'))
        model.add(layers.Dropout(0.3))
        model.add(layers.Dense(1,activation = 'sigmoid'))
        model.compile(optimizer = 'Adam',
                      loss = 'binary_crossentropy',
                      metrics = ['acc'])
        model.summary()

        # Train a model
        history = model.fit(binary_partial_train_data,
                            binary_partial_train_labels,
                            validation_data = (binary_val_data,
                                               binary_val_labels),
                            epochs = 10, batch_size = 18, verbose = 1)

        # Shows the acc and loss plot
        fig, loss_ax = plt.subplots()
        acc_ax = loss_ax.twinx()

        plt.title('Training and validation acc and loss')
        loss_ax.plot(history.history['loss'],'y',label = 'train loss')
        loss_ax.plot(history.history['val_loss'],'r',label = 'val loss')

        acc_ax.plot(history.history['acc'],'b',label = 'train acc')
        acc_ax.plot(history.history['val_acc'],'g',label = 'val acc')

        loss_ax.set_xlabel('epoch')
        loss_ax.set_ylabel('loss')
        acc_ax.set_ylabel('accuracy')

        loss_ax.legend(loc = 'upper left')
        acc_ax.legend(loc = 'lower left')
        plt.show()

        # Test a model
        print("-- Evaluate --")
        score = model.evaluate(test_data,test_labels,verbose=1)
        print('loss :', score[0])
        print("%s : %.2f%%" %(model.metrics_names[1],score[1]*100))
        print("computation_time :","%.2fs" %(time.time() - preprocessing))
        print('#################_Binary_classification_completed_##########')

class Interface(Extra):
    """
    Interface module
    * preprocessing_interface : Set the initial data input points.
    * hyperparameter_interface : Tuning the model parameters.
    * extra_image_interface : Run the extra modules.

    Author : JiHoon Kim
    Last-modified : 13th July 2020
    """
    def preprocessing_interface(self,data):
        """
        This function for setting the input data,
        preprocessing the feature, and loading the labels.
        data : 'Y' use the default subject-3.
        """
        folder1 = 'acquisition'
        folder2 = 'preprocessing'

        # Use the defualt: subject-3 data
        if 'Y'==data:
            input_file = ['ds000105/sub-3/func/bold.nii.gz']
            output_name = ['sub-3_bold.nii.gz']
            mask_file = ['ds000105/sub-3/mask/mask4_vt.nii.gz']
            input_csv = 'ds000105/sub-3/task_events.csv'

        # Select the subject nubmer between 1 to 6.
        else:
            number = input("Please type new subject number betweeen 1 to 6: ")
            input_file = ['ds000105/sub-'+str(number)+'/func/bold.nii.gz']
            output_name = ['sub-'+str(number)+'_bold.nii.gz']
            mask_file = ['ds000105/sub-'+str(number)+'/mask/mask4_vt.nii.gz']
            input_csv = 'ds000105/sub-'+str(number)+'/task_events.csv'

        # Proceed the preprocessing features and labels
        prep_file = func.acquisition(folder1,input_file,output_name)
        mcflirt_file = func.mcflirt(folder2,prep_file)
        func_data = func.niftimasker(mcflirt_file,mask_file)
        numbers_list,labels_list = func.categorical_label(input_csv)
        return func_data,numbers_list,labels_list
        
    def hyperparameter_interface(self):
        """
        This function for tuning the hpyerparameter data.
        """

        # set the hyperparameter
        do_rate = input("Please type the drop_out rate : ")
        n1 = input("Please type the neuron layers 1 : ")
        n2 = input("Please type the neuron layers 2 : ")
        cv_parameter = [float(do_rate),int(n1),int(n2)]
        return cv_parameter
  
    def extra_image_interface(self,data):
        """
        This function for setting the input data,
        creating a mask image, displaying images.
        data : 'Y' use the default subject-3.
        """
        folder3 = 'mask'

        # Use the defualt: subject-3 data
        if 'Y'==data:
            mask_in_path = ['ds000105/sub-3/anat/anat.nii.gz']
            mask_out_path = ['sub-3_anat']
            image = ['ds000105/sub-3/anat/anat.nii.gz',
                     'mask/sub-3_anat.nii.gz','mask/sub-3_anat_mask.nii.gz',
                     'ds000105/sub-3/mask/mask4_vt.nii.gz']
            title = ['sub-3_anat','sub-3_WM','sub-3_mask','sub-3_vt']
            func.func_img('acquisition/sub-3_bold.nii.gz','original',
                          'motion correction','preprocessing/sub-3_bold.nii.gz')

        # Select the subject nubmer between 1 to 6.
        else:
            number2 = input("Please type the subject number betweeen 1 to 6: ")
            mask_in_path = ['ds000105/sub-'+str(number2)+'/anat/anat.nii.gz']
            mask_out_path = ['sub-'+str(number2)+'_anat']
            image = ['ds000105/sub-'+str(number2)+'/anat/anat.nii.gz',
                     'mask/sub-'+str(number2)+'_anat.nii.gz',
                     'mask/sub-'+str(number2)+'_anat_mask.nii.gz',
                     'ds000105/sub-'+str(number2)+'/mask/mask4_vt.nii.gz']
            title = ['sub-'+str(number2)+'_anat','sub-'+str(number2)+'_WM',
                     'sub-'+str(number2)+'_mask','sub-'+str(number2)+'_vt']
            func.func_img('acquisition/sub-'+str(number2)+'_bold.nii.gz',
                          'original','motion correction',
                          'preprocessing/sub-'+str(number2)+'_bold.nii.gz')                     
        func.mask_image(folder3,mask_in_path,mask_out_path)
        func.anat_mask(image,title)

##INSTRUCTIONS_STARTED##########################################################
print('Welcome to neural network programming project interface')
func = Interface()

##MAIN_ANALYSIS:PREPROCESSING_TO_GRIDSEARCH#####################################
print('#################_Analysis_started_#########################')
programming = time.time()

# Choose the default or other subject between 1 to 6
choice = input("Do you want to analysis Subject 3 or not? Type Y or N : ")

# Preprocess the feature and load the list
func_data,numbers_list,labels_list = func.preprocessing_interface(choice)

# Prepare the dataset
train_data,test_data,train_labels,test_labels,size = func.categorical_dataset(
    labels_list,func_data,numbers_list)

# Run the gridsearch
func.grid_search(train_data,train_labels)

##MAIN_ANALYSIS:CROSS_VALIDATION_TO_TRAIN_AND_TEST##############################
# Run the cross validation k = 5
batch_size = input("Please type the batch size : ")
cv_parameter = func.hyperparameter_interface()
func.cross_validation(int(batch_size),cv_parameter,train_data,train_labels)

# Train and test the model
for final_epochs in [25,50,75,100,200]:
    func.test_model(final_epochs,cv_parameter)
print("total_computation_time :","%.2fs" %(time.time() - programming))
print('#################_Analysis_completed_#######################')

##EXTRA_ANALYSIS:MASK_AND_SHOW_IMAGES###########################################
print('#################_Analysis_extra_started####################')
programming2 = time.time()

# Choose the default or other subject between 1 to 6 based on previous analysis
choice = input("Did you test subject 3 or not? Type Y or N : ")

# Mask the image and shows the images: fMRI data signal, and mask iamge
func.extra_image_interface(choice)

##EXTRA_ANALYSIS:BINARY_CLASSIFICATION##########################################
# Do the binary classification and shows the result with model
func.binary_classification(func_data,labels_list,numbers_list)
print("total_computation_time :","%.2fs" %(time.time() - programming2))
print('#################_Analysis_extra_completed##################')
##INSTRUCTIONS_COMPLETED########################################################