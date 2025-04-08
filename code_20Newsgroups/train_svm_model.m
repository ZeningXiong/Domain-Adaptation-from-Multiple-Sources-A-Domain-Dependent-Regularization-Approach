% filepath: f:\港大sem2\COMP7404\group project\7404_p\7404_p\code_20Newsgroups\train_svm_model.m
function train_svm_model(data_file)
    try
        % Add LIBSVM path
        addpath('F:\MATLAB\libsvm-3.35\libsvm-3.35\matlab');
        disp('LIBSVM path added');
        
        % Remove extra quotes from the path if any
        data_file = strrep(data_file, '"', '');
        
        disp(['Current working directory: ', pwd]);
        disp(['Data file: ', data_file]);
        
        % Check if file exists
        if ~exist(data_file, 'file')
            error('Data file not found: %s', data_file);
        end
        
        % Load data
        disp('Loading data...');
        data = load(data_file);
        
        % Convert data types
        tilde_y = double(data.tilde_y);
        K_v = double(data.K_v);
        v_ind = double(data.v_ind);
        tar_index = double(data.tar_index);
        K = double(data.K);
        C = double(data.C);
        epsilon = double(data.epsilon);
        
        % Correct dimensions: ensure tilde_y is a column vector
        if size(tilde_y, 2) > 1 && size(tilde_y, 1) == 1
            disp('Converting tilde_y to column vector...');
            tilde_y = tilde_y';
        end
        
        % Display data dimensions
        disp(['tilde_y dimensions: ', num2str(size(tilde_y,1)), 'x', num2str(size(tilde_y,2))]);
        disp(['K_v dimensions: ', num2str(size(K_v,1)), 'x', num2str(size(K_v,2))]);
        
        % Check dimension matching
        if size(tilde_y, 1) ~= size(K_v, 1)
            error('Label vector length (%d) does not match the number of instances (%d)', size(tilde_y, 1), size(K_v, 1));
        end
        
        % Train SVM model
        disp('Running svmtrain...');
        instance_indices = (1:size(K_v,1))';
        
        % Display detailed training information
        disp(['First sample label: ', num2str(tilde_y(1))]);
        disp(['Label type: ', class(tilde_y)]);
        disp(['Instance type: ', class(instance_indices)]);
        disp(['Kernel matrix type: ', class(K_v)]);
        
        % LIBSVM parameters
        param_str = sprintf('-s 3 -c %g -t 4 -p %g', C, epsilon);
        disp(['SVM parameters: ', param_str]);
        
        % Train the model
        model = svmtrain(tilde_y, [instance_indices, K_v], param_str);
        
        % Check the model
        if ~isfield(model, 'SVs') || isempty(model.SVs)
            error('Model training failed: No support vectors found');
        end
        
        % Map support vectors to original indices
        disp('Mapping support vectors...');
        if length(model.SVs) > length(v_ind)
            error('Support vector index out of range: SV=%d, v_ind=%d', length(model.SVs), length(v_ind));
        end
        
        SVs_mapped = v_ind(model.SVs);
        
        % Calculate decision values
        disp('Calculating decision values...');
        dv = K(tar_index, SVs_mapped) * model.sv_coef - model.rho;
        
        % Display result information
        disp(['Number of support vectors: ', num2str(length(model.SVs))]);
        disp(['Decision values dimensions: ', num2str(size(dv,1)), 'x', num2str(size(dv,2))]);
        
        % Save results
        disp(['Saving results to: ', data_file]);
        save(data_file, 'model', 'dv', '-append');
        disp('SVM training completed');
        
    catch ME
        disp('==== Error Details ====');
        disp(['Error message: ', ME.message]);
        for i = 1:length(ME.stack)
            disp(['File: ', ME.stack(i).file]);
            disp(['Function: ', ME.stack(i).name]);
            disp(['Line: ', num2str(ME.stack(i).line)]);
        end
        disp('======================');
        exit(1);  % Exit with error
    end
    exit(0);  % Normal exit
end
