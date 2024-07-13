% config.m
addpath("activation_functions")

% Random matrix columns
W1_c = [16 42 100];

% Activation functions to test
activation_functions = {@identity, @sigmoid, @relu, @tanh};
activation_functions_names = {'identity', 'sigmoid', 'relu', 'tanh'};