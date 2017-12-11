clc;clear all;close all;
%% 
mkdir data;
rank_size = 2000;
%% network name
netname = 'ResNet_50'; % network: CaffeNet  or ResNet_50
%% add necessary paths
query_dir = 'dataset/query/';% query directory
test_dir = 'dataset/bounding_box_test/';% database directory
train_dir = 'dataset/bounding_box_train/';% training directory
%% calculate query features
Hist_query = importdata(['test/IDE_' netname '_query_market2duke_AGAIN_pool5.mat']);
% Hist_query = importdata(['feat/IDE_' netname '_baseline_train_stage2_SI_pool5.mat']);
% Hist_query = importdata('feat_multi/LOVE_Hist_query_max.mat');

nQuery = size(Hist_query, 2);

%% calculate database features
Hist_test = importdata(['test/IDE_' netname '_test_market2duke_AGAIN_pool5.mat']);
nTest = size(Hist_test, 2);

%% normalize
sum_val = sqrt(sum(Hist_test.^2));
for n = 1:size(Hist_test, 1)
    Hist_test(n, :) = Hist_test(n, :)./sum_val;
end

sum_val = sqrt(sum(Hist_query.^2));
for n = 1:size(Hist_query, 1)
    Hist_query(n, :) = Hist_query(n, :)./sum_val;
end

% sum_val = sqrt(sum(Hist_train.^2));
% for n = 1:size(Hist_train, 1)
%     Hist_train(n, :) = Hist_train(n, :)./sum_val;
% end
%% calculate the ID and camera for training images
train_files = dir([train_dir '*.jpg']);
trainID = zeros(length(train_files), 1);
trainCAM = zeros(length(train_files), 1);
if ~exist('data/trainID_duke.mat')
    for n = 1:length(train_files)
        img_name = train_files(n).name;
        if strcmp(img_name(1), '-') % junk images
            trainID(n) = -1;
            trainCAM(n) = str2num(img_name(5));
        else
            %img_name
            trainID(n) = str2num(img_name(1:4));
            trainCAM(n) = str2num(img_name(7));
        end
    end
    save('data/trainID_duke.mat', 'trainID');
    save('data/trainCAM_duke.mat', 'trainCAM');
else
    trainID = importdata('data/trainID_duke.mat');
    trainCAM = importdata('data/trainCAM_duke.mat');    
end
%% calculate the ID and camera for database images
test_files = dir([test_dir '*.jpg']);
testID = zeros(length(test_files), 1);
testCAM = zeros(length(test_files), 1);
if ~exist('data/testID_duke.mat')
    for n = 1:length(test_files)
        img_name = test_files(n).name;
        if strcmp(img_name(1), '-') % junk images
            testID(n) = -1;
            testCAM(n) = str2num(img_name(5));
        else
            %img_name
            testID(n) = str2num(img_name(1:4));
            testCAM(n) = str2num(img_name(7));
        end
    end
    save('data/testID_duke.mat', 'testID');
    save('data/testCAM_duke.mat', 'testCAM');
else
    testID = importdata('data/testID_duke.mat');
    testCAM = importdata('data/testCAM_duke.mat');    
end

%% calculate the ID and camera for query images
query_files = dir([query_dir '*.jpg']);
queryID = zeros(length(query_files), 1);
queryCAM = zeros(length(query_files), 1);
if ~exist('data/queryID_duke.mat')
    for n = 1:length(query_files)
        img_name = query_files(n).name;
        if strcmp(img_name(1), '-') % junk images
            queryID(n) = -1;
            queryCAM(n) = str2num(img_name(5));
        else
            queryID(n) = str2num(img_name(1:4));
            queryCAM(n) = str2num(img_name(7));
        end
    end
    save('data/queryID_duke.mat', 'queryID');
    save('data/queryCAM_duke.mat', 'queryCAM');
else
    queryID = importdata('data/queryID_duke.mat');
    queryCAM = importdata('data/queryCAM_duke.mat');    
end

%% search the database and calcuate re-id accuracy
ap = zeros(nQuery, 1); % average precision
ap_max_rerank  = zeros(nQuery, 1); % average precision with MultiQ_max + re-ranking 
ap_pairwise = zeros(nQuery, 6); % pairwise average precision with single query (see Fig. 7 in the paper)

CMC = zeros(nQuery, rank_size);
CMC_max_rerank = zeros(nQuery, rank_size);

r1 = 0; % rank 1 precision with single query
r1_max_rerank = 0; % rank 1 precision with MultiQ_max + re-ranking
r1_pairwise = zeros(nQuery, 6);% pairwise rank 1 precision with single query (see Fig. 7 in the paper)

% my_pdist2 = @(A, B) sqrt( bsxfun(@plus, sum(A.^2, 2), sum(B.^2, 2)') - 2*(A*B'));
% dist = my_pdist2(Hist_test', Hist_query');
dist = sqdist(Hist_test, Hist_query); % distance calculate with single query. Note that Euclidean distance is equivalent to cosine distance if vectors are l2-normalized
%dist_cos_max = (2-dist_max)./2; % cosine distance with MultiQ_max, used for re-ranking

knn = 1; % number of expanded queries. knn = 1 yields best result

for k = 1:nQuery
    % load ground truth for each query (good and junk)
    good_index = intersect(find(testID == queryID(k)), find(testCAM ~= queryCAM(k)))';% images with the same ID but different camera from the query
    junk_index1 = find(testID == -1);% images neither good nor bad in terms of bbox quality
    junk_index2 = intersect(find(testID == queryID(k)), find(testCAM == queryCAM(k))); % images with the same ID and the same camera as the query
    junk_index = [junk_index1; junk_index2]';
    score = dist(:, k);
    
    % sort database images according Euclidean distance
    [~, index] = sort(score, 'ascend');  % single query
    
    % re-rank  select rank_size=1000 index
    index = index(1:rank_size);    
    
    [ap(k), CMC(k, :)] = compute_AP(good_index, junk_index, index);% compute AP for single query
%     fprintf('%d::%f\n',k,ap(k));
end
CMC = mean(CMC);
%% print result
fprintf('single query:                                    mAP = %f, r1 precision = %f\r\n', mean(ap), CMC(1));
%[ap_CM, r1_CM] = draw_confusion_matrix(ap_pairwise, r1_pairwise, queryCam);
%fprintf('average of confusion matrix with single query:  mAP = %f, r1 precision = %f\r\n', (sum(ap_CM(:))-sum(diag(ap_CM)))/30, (sum(r1_CM(:))-sum(diag(r1_CM)))/30);
% 
% %% re-ranking setting
% k1 = 20;
% k2 = 6;
% lambda = 0.3;
% %% Euclidean + re-ranking
% query_num = size(Hist_query, 2);
% dist_eu_re = re_ranking( [Hist_query Hist_test], 1, 1, query_num, k1, k2, lambda);
% [CMC_eu_re, map_eu_re, ~, ~] = evaluation(dist_eu_re,  testID, queryID,  testCAM, queryCAM);
% 
% fprintf(['The IDE (' netname ') + Euclidean + re-ranking performance:\n']);
% fprintf(' Rank1,  mAP\n');
% fprintf('%5.2f%%, %5.2f%%\n\n', CMC_eu_re(1) * 100, map_eu_re(1)*100);


%% plot CMC curves
% figure;
% s = 50;
% CMC_curve = CMC ;
% plot(1:s, CMC_curve(:, 1:s));

%% train and test XQDA
% [train_sample1, train_sample2, label1, label2] = gen_train_sample_xqda(trainID, trainCAM, Hist_train); % generate pairwise training features for XQDA
% [W, M_xqda] = XQDA(train_sample1, train_sample2, label1, label2);% train XQDA
% % Calculate distance
% dist_xqda = MahDist(M_xqda, Hist_test' * W, Hist_query' * W); % calculate MahDist between query and gallery boxes with learnt subspace. Smaller distance means larger similarity
% [CMC_xqda, map_xqda, ~, ~] = evaluation(dist_xqda, testID, queryID, testCAM, queryCAM);
% 
% fprintf(['The IDE (' netname ') + XQDA performance:\n']);
% fprintf(' Rank1,  mAP\n');
% fprintf('%5.2f%%, %5.2f%%\n\n', CMC_xqda(1) * 100, map_xqda(1)*100);
