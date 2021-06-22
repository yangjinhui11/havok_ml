% Yangjinhui
clear all, close all, clc
addpath('./utils');
% generate Data
sigma = 10;  % Lorenz's parameters (chaotic)
beta = 8/3;
rho = 28;
n = 3;
x0=[-8; 8; 27];  % Initial condition

% Integrate
dt = 0.01;
tspan=[dt:dt:200];
N = length(tspan);
options = odeset('RelTol',1e-12,'AbsTol',1e-12*ones(1,n));
[t,xdat]=ode45(@(t,x) lorenz(t,x,sigma,beta,rho),tspan,x0,options);
save ./DATA/lorenz.mat