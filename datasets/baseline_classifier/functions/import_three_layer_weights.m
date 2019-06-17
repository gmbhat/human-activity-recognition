function [W1, W2, W3] = import_three_layer_weights(filepath, N_inputs, N1, N2, N3)
% Function to read the weights of a three layer network (2 hidden, 1
% output) from the csv file
% Inputs:
% filepath: Path of the csv file
% N_inputs: Number of inputs to the network
% N1: Number of neurons in the first hidden layer
% N2: Number of neurons in the second hidden layer
% N3: Number of neurons in the output layer

% Outputs:
% W1, W2, W2: Weights of Layer 1, layer 2 and 3, respectively

%% Read the weights raw file
model_weights = importfile_weights(filepath);

% Extract weights of the first layer
W1 = model_weights(1:N_inputs, 1:N1);
W1_bias = model_weights(N_inputs+1:N_inputs+N1, 1);

% Extract the weights of the second layer
W2 = model_weights(N_inputs+N1+1:N_inputs+N1+N1, 1:N2);
W2_bias = model_weights(N_inputs+N1+N1+1:N_inputs+N1+N1+N2, 1);

W3 = model_weights(N_inputs+N1+N1+N2+1:N_inputs+N1+N1+N2+N2, 1:N3);
W3_bias = model_weights(N_inputs+N1+N1+N2+N2+1:N_inputs+N1+N1+N2+N2+N3, 1);

% Merge W1 weights and the bias term
W1 = [W1_bias'; W1];

% Merge W2 weights and the bias term
W2 = [W2_bias'; W2];

% Merge W2 weights and the bias term
W3 = [W3_bias'; W3];