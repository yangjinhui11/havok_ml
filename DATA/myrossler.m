% Copyright 2016, All Rights Reserved
% Code by Steven L. Brunton
clear all, close all, clc
figpath = './figures/';
addpath('./utils');

% generate Data
a = .1;  
b = .1;
c = 14;
n = 3;
x0=[1; 1; 1];  % Initial condition

% Integrate
dt = 0.01;
tspan=[dt:dt:500];
N = length(tspan);
options = odeset('RelTol',1e-12,'AbsTol',1e-12*ones(1,n));
[t,xdat]=ode45(@(t,x) rossler(t,x,a,b,c),tspan,x0,options);
save ./DATA/rossler.mat