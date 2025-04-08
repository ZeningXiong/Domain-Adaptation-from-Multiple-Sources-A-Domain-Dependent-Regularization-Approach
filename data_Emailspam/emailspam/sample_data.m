function sample_data(nRound)

%### generate the partition for experiments
datapath = '.';
nSample = 50;
rand('seed', 123); % we fix the randseed to ensure that the experiments can be exactly repeated.
for d = 1:3
    load(fullfile(datapath, ['data_', num2str(d), '.mat']), 'features', 'labels');
    pos_ind = find(labels == 1);
    neg_ind = find(labels == -1);
    for r = 1:nRound
        pos_tmp = randperm(length(pos_ind));
        pos_tmp = pos_tmp(:);
        neg_tmp = randperm(length(neg_ind));
        neg_tmp = neg_tmp(:);
        
        tar_training_ind = [pos_ind(pos_tmp(1:nSample)); neg_ind(neg_tmp(1:nSample))];
        test_ind = setdiff(1:length(labels), tar_training_ind);
        tar_background_ind = test_ind;
        
        if ~exist(fullfile('tar_ind', ['data_', num2str(d)]), 'dir')
            mkdir(fullfile('tar_ind', ['data_', num2str(d)]));
        end
        save(fullfile('tar_ind', ['data_', num2str(d)], [num2str(r), '.mat']), 'tar_training_ind', 'test_ind', 'tar_background_ind');
    end    
    clear features labels
end
