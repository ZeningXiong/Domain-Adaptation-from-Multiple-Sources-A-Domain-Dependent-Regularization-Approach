% filepath: f:\港大sem2\COMP7404\group project\7404_p\7404_p\code_Emailspan\train_svm_model.m
function train_svm_model(data_file)
    try
        % Add LIBSVM path
        addpath('F:\MATLAB\libsvm-3.35\libsvm-3.35\matlab');
        disp('LIBSVM path added');
        
        % Remove extra quotes from the path
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
        tilde_y = double(data.tilde_y(:));  % Ensure column vector
        v_ind = double(data.v_ind(:));        % Ensure column vector
        tar_index = double(data.tar_index(:));% Ensure column vector
        K = double(data.K);
        add_kernel = double(data.add_kernel(:));  % Ensure column vector
        C = double(data.C);
        epsilon = double(data.epsilon);

        % Display dimensions of data
        disp(['tilde_y dimension: ', num2str(length(tilde_y)), ' elements']);
        disp(['v_ind dimension: ', num2str(length(v_ind)), ' elements']);
        disp(['v_ind maximum value: ', num2str(max(v_ind))]);
        disp(['v_ind minimum value: ', num2str(min(v_ind))]);
        
        % Check index range and adjust tilde_y if necessary
        if max(v_ind) > length(tilde_y)
            disp(['Warning: v_ind maximum value (', num2str(max(v_ind)), ') exceeds tilde_y length (', num2str(length(tilde_y)), ')']);
            disp('Expanding tilde_y to accommodate all indices');
            new_tilde_y = zeros(max(v_ind), 1);
            new_tilde_y(1:length(tilde_y)) = tilde_y;
            tilde_y = new_tilde_y;
            
            new_add_kernel = ones(max(v_ind), 1);
            new_add_kernel(1:length(add_kernel)) = add_kernel;
            add_kernel = new_add_kernel;
            
            disp(['Adjusted tilde_y dimension: ', num2str(length(tilde_y))]);
        end
        
        % Extract labels corresponding to v_ind
        disp('Extracting labels for v_ind...');
        tilde_y_v = tilde_y(v_ind);
        
        % Prepare weighted kernel matrix
        disp('Preparing weighted kernel matrix...');
        K_v = K(v_ind, v_ind) + diag(add_kernel(v_ind));
        
        % Display dimensions of prepared data
        disp(['tilde_y_v dimension: ', num2str(length(tilde_y_v)), ' elements']);
        disp(['K_v dimensions: ', num2str(size(K_v,1)), 'x', num2str(size(K_v,2))]);
        
        % Check dimension match
        if length(tilde_y_v) ~= size(K_v, 1)
            error('Label vector length (%d) does not match number of instances (%d)', length(tilde_y_v), size(K_v, 1));
        end
        
        % Train SVM model
        disp('Executing svmtrain...');
        instance_indices = (1:length(tilde_y_v))';
        
        % LIBSVM parameters - remove -q parameter to be consistent with original implementation
        param_str = sprintf('-s 3 -c %g -t 4 -p %g', C, epsilon);
        disp(['SVM parameters: ', param_str]);
        
        % Train the model
        model = svmtrain(tilde_y_v, [instance_indices, K_v], param_str);
        
        % Check model
        if ~isfield(model, 'SVs') || isempty(model.SVs)
            error('Model training failed: Support vectors not found');
        end
        
        % Map support vectors to original indices
        disp('Mapping support vectors...');
        if max(model.SVs) > length(v_ind)
            error('Support vector index out of range: SV maximum value=%d, v_ind length=%d', max(model.SVs), length(v_ind));
        end
        
        % Modification: Use sparse function to process support vectors, consistent with original implementation
        SVs_mapped = sparse(v_ind(model.SVs));
        
        % Calculate decision values
        disp('Calculating decision values...');
        dv = K(tar_index, SVs_mapped) * model.sv_coef - model.rho;
        
        % Display result information
        disp(['Number of support vectors: ', num2str(length(model.SVs))]);
        disp(['Decision values dimension: ', num2str(length(dv)), ' elements']);
        
        % Save results
        disp(['Saving results to: ', data_file]);
        save(data_file, 'model', 'dv', 'SVs_mapped', '-append');
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