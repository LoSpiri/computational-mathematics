% config.m
addpath("activation_functions")

% Random matrix columns
% This corresponds to W1_c and W2_r
% k = [16 42 100];
k = 16;
random_state = 42;

% Activation functions to test
% activation_functions = {@identity, @sigmoid, @relu, @tanh};
% activation_functions_names = {'identity', 'sigmoid', 'relu', 'tanh'};
activation_functions = {@tanh};
activation_functions_names = {'tanh'};

% delta = [0.25 0.5 0.75];
% rho = [0.5 0.7 0.9];
% R = [10 20 50];
% max_iter = 1000;
% lambda = [1e-3 1e-4 1e-5];

delta = [0.01];
rho = [0.95];
R = [5];
max_iter = 1000;
lambda = [1e-4];