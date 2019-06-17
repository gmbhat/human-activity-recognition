function [transition_label, prob_test, output_in] = classify_har_data_three_layers(feature_vector, W1, W2, W3)
m = size(feature_vector, 1);

hidden_layer_y = ([ones(m, 1) feature_vector] * W1);

hidden_layer_y = relu_activation(hidden_layer_y);


% Hidden layer 2
% Apply weights of the output layer
hidden_layer_2_in = [ones(m, 1) hidden_layer_y];

hidden_layer_2_y = hidden_layer_2_in* W2;
hidden_layer_2_y = relu_activation(hidden_layer_2_y);


output_in = [ones(m, 1) hidden_layer_2_y];
output_layer_y = (output_in * W3);
    
[transition_label, prob_test] = softmax_classification(output_layer_y);

end % function transition_label = classify_har_data(feature_vector, W1, W2)