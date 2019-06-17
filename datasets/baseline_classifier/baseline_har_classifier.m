% This file implements the baseline neural network classifier for HAR
% It takes the feature matrices as an input and output the classification
% for each feature window

clear variables;
close all;
addpath('functions');
% Set random number generator
rng(1);

global sensortag;
global stretch;

% Setup indices for sensortag and stretch
sensortag.user_index      = 1;
sensortag.scenario_index  = 2;
sensortag.trial_index     = 3;
sensortag.foot_index      = 4;
sensortag.fs_index        = 5;
sensortag.seq_index       = 6;
sensortag.wall_time_index = 7;
sensortag.time_index      = 8;
sensortag.fifo_index      = 9;
sensortag.ax_index        = 10;
sensortag.ay_index        = 11;
sensortag.az_index        = 12;
sensortag.gyrX_index      = 13;
sensortag.gyrY_index      = 14;
sensortag.gyrZ_index      = 15;

% Indices for stretch
stretch.user_index      = 1;
stretch.scenario_index  = 2;
stretch.trial_index     = 3;
stretch.wall_time_index = 4;
stretch.channel1_index  = 5;
stretch.channel2_index  = 6;
stretch.channel3_index  = 7;
stretch.channel4_index  = 8;
stretch.channel5_index  = 9;
stretch.channel6_index  = 10;
stretch.channel7_index  = 11;
stretch.channel8_index  = 12;
stretch.channel9_index  = 13;
stretch.channel10_index = 14;

% possible_merged_labels = {'drive2drive';'jump2jump';'lie2lie';...
%     'sit2sit';'stand2stand';'walk2walk';'transition';'undefined'; 'stairsup';'stairsdown'};

% Jump = 1
% Lie down = 2
% Sit = 3
% Stand = 4
% Walk = 5
% Stairsup = 6
% Stairs down = 7
% Transition  = 8
% Drive  = 9
% Undefined = 10
possible_merged_labels = {'jump2jump';'lie2lie';...
    'sit2sit';'stand2stand';'walk2walk';'stairsup';'stairsdown';'transition';'drive2drive'; 'undefined'};

undefined_index = find(ismember(possible_merged_labels, 'undefined'));
sit_activity_index = find(ismember(possible_merged_labels, 'sit-activity'));
drive_index = find(ismember(possible_merged_labels, 'drive2drive'));

% First find the unique labels

% Variable to track real number of features
real_feature_num = 1;

%% Load the parameters related to neural network
% Load the indices for which training and test was done in Python training
load('features_file.mat');

% Indices of features to normalize
features_to_normalize = 1:119;                                              % Choose 1 to 119 because we don't want to normalize prev label and activity time
features_to_use = 1:120;
%% Read the weights../python_code/model_files/gb_test/model_final_scaled_1_6.csv
% Define inputs, neurons and outputs
N_inputs = 120;                                                              % number of features generated. 117 for 16 FFT-co-eff, 165 for 64 FFT-co-eff
N1 = 4;
N2 = 8;
N3 = 8;

weight_file1 = 'weights.csv';
% Load the weights
[W11, W21, W31] = import_three_layer_weights(weight_file1, N_inputs, N1, N2, N3);

%% Variables to store the prediction results

%% Perform the analysis here
nn_classifier_results = classify_har_data_three_layers(feature_matrix_norm(:,features_to_use), W11, W21, W31); 
        
% First get the confusion matrix
C = confusionmat(label_vector,nn_classifier_results);

C_train = confusionmat(label_vector(train_idx), nn_classifier_results(train_idx));
C_xval = confusionmat(label_vector(xval_idx), nn_classifier_results(xval_idx));
C_test = confusionmat(label_vector(test_idx), nn_classifier_results(test_idx));

sum_C = sum(C,2);

accuracy_state = (C./sum_C)*100;

% Overall difference
overall_difference = label_vector - nn_classifier_results;

% Calculate the train and test accuracies
train_differnce = label_vector(train_idx) - nn_classifier_results(train_idx);          % Difference between ref and prediction for the training indices

% Xval difference
xval_difference = label_vector(xval_idx) - nn_classifier_results(xval_idx);

% Test difference
test_difference = label_vector(test_idx) - nn_classifier_results(test_idx);

overall_accuracy = 100 - (nnz(overall_difference)/length(overall_difference))*100;            % Get the overall accuracy
train_accuracy = 100 - (nnz(train_differnce)/length(train_differnce))*100;            % Get the train accuracy
test_accuracy = 100 - (nnz(test_difference)/length(test_difference))*100;            % Get the test accuracy
xval_accuracy = 100 - (nnz(xval_difference)/length(xval_difference))*100;            % Get the xval accuracy

fprintf('Overall accuracy = %0.2f%% Train accuracy = %0.2f%%, Xval accuracy = %0.2f%% Test accuracy = %0.2f%%\n', overall_accuracy, ...
    train_accuracy, xval_accuracy, test_accuracy);