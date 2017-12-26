clc;clear all;close all;
%***********************************************%
% This code runs on the Market-1501 dataset.    %
% Please modify the path to your own folder.    %
% We use the mAP and hit-1 rate as evaluation   %
%***********************************************%
% if you find this code useful in your research, please kindly cite our
% paper as,
% Liang Zheng, Liyue Sheng, Lu Tian, Shengjin Wang, Jingdong Wang, and Qi Tian,
% Scalable Person Re-identification: A Benchmark, ICCV, 2015.

% Please download Market-1501 dataset and unzip it in the "dataset" folder.
addpath(genpath('LOMO_XQDA/'));
addpath(genpath('utils/'));
addpath(genpath('KISSME/'));
run('KISSME/toolbox/init.m');

%% network name
netname = 'ResNet_50'; % network: CaffeNet  or ResNet_50

%% test info
galFea = importdata(['feat/IDE_' netname '_test.mat']);
galFea = double(galFea);
probFea = importdata(['feat/IDE_' netname '_query.mat']);
probFea = double(probFea);
label_gallery = importdata('data/testID.mat');
label_query = importdata('data/queryID.mat');
cam_gallery =   importdata('data/testCam.mat');
cam_query =  importdata('data/queryCam.mat');

%% normalize
sum_val = sqrt(sum(galFea.^2));
for n = 1:size(galFea, 1)
    galFea(n, :) = galFea(n, :)./sum_val;
end

sum_val = sqrt(sum(probFea.^2));
for n = 1:size(probFea, 1)
    probFea(n, :) = probFea(n, :)./sum_val;
end

% Instead of pdist2. 50 times faster than pdist2
my_pdist2 = @(A, B) sqrt( bsxfun(@plus, sum(A.^2, 2), sum(B.^2, 2)') - 2*(A*B'));
%% Euclidean

dist_eu = my_pdist2(galFea', probFea');
[CMC_eu, map_eu, ~, ~] = evaluation(dist_eu, label_gallery, label_query, cam_gallery, cam_query);

fprintf(['The IDE (' netname ') + Euclidean performance:\n']);
fprintf(' Rank1,  mAP\n');
fprintf('%5.2f%%, %5.2f%%\n\n', CMC_eu(1) * 100, map_eu(1)*100);

