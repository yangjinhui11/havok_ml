clear all, close all, clc
figpath = './figures/';
addpath('./utils');

gamma = 0.1;
beta = 0.2;
tau = 17;
n = 10;

% Integrate
dt = .1;
tspan = [0:dt:5500];
options = ddeset('MaxStep',dt);
sol=dde23(@(t,y,ytau) MackeyGlass(t,y,ytau,gamma,beta,n),tau,.8,tspan,options);
t = sol.x;
xdat = (sol.y)';

save ./DATA/mackeyglass.mat