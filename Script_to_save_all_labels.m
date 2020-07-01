clc;clear all;close all;
load("wiki.mat")
y = wiki.gender;
save("labels.mat",'y');
disp("Done!")