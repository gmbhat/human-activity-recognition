function output = relu_activation(input)
output = input;
output_n_indices = output < 0;
output(output_n_indices) = 0;
end