%% Etract IDE features
clear;clc;
%addpath('../caffe/matlab/');
addpath(genpath('utils/'));
% load model and creat network
caffe.set_device(1);
caffe.set_mode_gpu();

netname = 'ResNet_50'; % network: CaffeNet  or ResNet_50

batch_size=16;
% set your path to the prototxt and model
model =  ['../models/duke/' netname '/' netname '_test.prototxt'];
% weights = ['../output/duke_train/IDE_ResNet_50_duke_max_iter_50000.caffemodel'];
% weights = ['../output/duke_train/IDE_ResNet_50_duke_bn_LSR_iter_50000.caffemodel'];
% weights = ['../output/market_train/IDE_ResNet_50_market_iter_50000.caffemodel'];
weights = ['../output/duke_train/IDE_ResNet_50_market2duke_AGAIN_iter_40000.caffemodel'];

net = caffe.Net(model, weights, 'test');

if strcmp(netname, 'CaffeNet')
    im_size = 227;
    feat_dim = 256;
elseif strcmp(netname, 'CaffeNet_source')
    im_size = 227;
    feat_dim = 4096;
else
    im_size = 224;
    feat_dim = 2048;
end
% mean data
mean_data = importdata('../weijian/matlab/+caffe/imagenet/ilsvrc_2012_mean.mat');
image_mean = mean_data;
off = floor((size(image_mean,1) - im_size)/2)+1;
image_mean = image_mean(off:off+im_size-1, off:off+im_size-1, :);

ef_path = {'dataset/bounding_box_train/', 'dataset/bounding_box_test/', 'dataset/query/'};
ef_name = {'train', 'test', 'query'};

if ~exist('feat')
    mkdir('feat')
end

% extract features
for i = 2:3
    %i = 2;
    img_path = ef_path{i};
    img_file = dir([img_path '*.jpg']);
    
    feat = [];
    feat1 = [];
    
    for n = 1:length(img_file)
        
        if mod(n, 4000) ==0
            fprintf('%s: %d/%d\n',ef_name{i}, n, length(img_file))
        end
        
        img_name = [img_path  img_file(n).name];
        im = imread(img_name);
        im = prepare_img( im, image_mean, im_size);
        
        input_data(:,:,:,mod(n-1,batch_size)+1)=im;
        input_blob={input_data};
        
        if(mod(n,batch_size)==0|| n==length(img_file))
            net.forward(input_blob);
            tmp0 = net.blobs('pool5').get_data();
            tmp = reshape(tmp0, 2048, 16);
            feat = [feat,tmp];

        end
        
        if (mod(n,2000)==0)
            count=(n/2000);
            save (['tmp_res/tmp_feature_pool5_',num2str(count),'.mat'],'feat');
            feat = [];
        end
        
        if (n==length(img_file))
            for tmp_count=count:-1:1
                group_feature = load(['tmp_res/tmp_feature_pool5_',num2str(tmp_count),'.mat']);
                
                feat = [group_feature.feat,feat];
                
            end
        end
    end
    
    feat = feat(:,1:length(img_file)); 
    save(['test/IDE_'  netname  '_' ef_name{i} '_market2duke_AGAIN_pool5.mat'], 'feat');
    feat = [];
    
end

caffe.reset_all();
evaluation_res_duke_fast;