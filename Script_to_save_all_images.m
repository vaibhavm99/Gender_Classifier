clc;clear all;close all;
load("wiki.mat")
x = wiki.full_path;
for i = 5140:62328
    str = append(num2str(i),'.mat');
    temp = x(i);
    a = char(temp);
    s = imread(a);
    save(str,'s');
    disp(append('Done ',num2str((i/62328)*100),'%'))
end
disp("Finished!");