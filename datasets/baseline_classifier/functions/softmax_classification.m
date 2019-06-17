function [pred_test, prob_test] = softmax_classification(input_data)

input_data_exp = exp(input_data);
prob_test = input_data_exp./sum(input_data_exp, 2);
[~, pred_test] = max(prob_test, [], 2);
end