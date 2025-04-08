function result = main_univerdam(data, C, N, lambda_L, lambda_D1, lambda_D2, thr, beta, virtual_label_type, kernel_types, kernel_params)
result_dir = sprintf('result_%s', data.setting);
result_file = fullfile(result_dir, ['result_', mfilename, '.txt']);
log_print(result_file, '<==========  BEGIN @ %s, C = %g, lambda_L = %g, lambda_D1 = %g, lambda_D2 = %g,  thr = %g, beta = %g============>\n', datestr(now), C, lambda_L, lambda_D1, lambda_D2, thr, beta);


X{1,1} = data.Xt;
y{1,1} = data.yt;
domain_index{1,1} = (1:length(data.yt))';
offset = length(data.yt);
for s = 1 : length(data.Xs)
    X{s+1,1} = data.Xs{s};
    y{s+1,1} = data.ys{s};
    domain_index{s+1,1} = (1:length(data.ys{s}))' + offset;
    offset = offset + length(data.ys);
end
X = cell2mat(X);
y = cell2mat(y);
tar_index = domain_index{1,1};
src_index = cell2mat(domain_index(2:end));
K = full(X*X');

for r = 1 : data.nRound
    tar_train_index = return_ind2(data.perm_tar_index{r}, sum(data.yt==1), N);
    tar_test_index = setdiff( (1:length(data.yt))', tar_train_index);
    
    all_test_dv = [];
    mmd_values = [];
    for s = 1 : length(data.Xs)
        if isequal(virtual_label_type(end-2:end), '_fr')
            dv_dir = fullfile(result_dir,  'decision_values','svm_fr', data.domain_names{s});
            mmd_dir = fullfile(result_dir, 'mmd_values_fr', data.domain_names{s});
        elseif isequal(virtual_label_type(end-1:end), '_s')
            dv_dir = fullfile(result_dir,  'decision_values', 'svm_s',data.domain_names{s});
            mmd_dir = fullfile(result_dir,'mmd_values_at', data.domain_names{s});
        elseif isequal(virtual_label_type(end-2:end), '_st')
            dv_dir = fullfile(result_dir, 'decision_values', 'svm_at', data.domain_names{s});
            mmd_dir = fullfile(result_dir,'mmd_values_at',  data.domain_names{s});
        end
        for kt = 1 : length(kernel_types)
            kernel_type = kernel_types{kt};
            for kp = 1 : length(kernel_params{kt})
                kernel_param = kernel_params{kt}(kp);
                if isequal(virtual_label_type(end-1:end), '_s')
                    dv_file = fullfile(dv_dir, sprintf('dv_C=%g_%s_%g.mat',C, kernel_type, kernel_param));
                else
                    dv_file = fullfile(dv_dir, sprintf('dv_round=%d_N=%d_C=%g_%s_%g.mat', r, N, C, kernel_type, kernel_param));
                end
                load(dv_file, 'decision_values');
                mmd_file = fullfile(mmd_dir, sprintf('mmd_%s_%g.mat', kernel_type, kernel_param));
                load(mmd_file, 'mmd_value');
                mmd_values = [mmd_values; mmd_value];
                all_test_dv = [all_test_dv, decision_values];
            end
        end
    end
    f_s = all_test_dv;
    pp = -beta * mmd_values.^2; %pp = pp - max(pp);
    gamma_s = exp(pp);
    gamma_s = gamma_s./sum(gamma_s);
    
    virtual_label = f_s * gamma_s;
    
    tilde_y = y;    tilde_y(src_index) = 0;
    tilde_y(tar_test_index) = virtual_label(tar_test_index);
    
    add_kernel = ones(length(tilde_y),1);
    add_kernel(src_index) = 1./lambda_D2;
    add_kernel(tar_train_index) = 1./lambda_L;
    add_kernel(tar_test_index) = 1./lambda_D1/sum(gamma_s);
    
    ind = find(abs(virtual_label(tar_test_index)) < thr);
    v_ind = [src_index; tar_train_index; setdiff(tar_test_index,tar_test_index(ind))];
    
    
    epsilon = 0.1;
    model = svmtrain(tilde_y(v_ind), [(1:length(v_ind))', K(v_ind, v_ind) + diag(add_kernel(v_ind))], sprintf('-s 3 -c %g -t 4 -q -p %g', C, epsilon));
    model.SVs = sparse(v_ind(model.SVs));
    dv = K(tar_index, model.SVs) * model.sv_coef - model.rho;
    
    ap = calc_ap(data.yt(tar_test_index), dv(tar_test_index));
    acc = mean(data.yt(tar_test_index)==sign(dv(tar_test_index)));
    result{r}.ap = ap;
    result{r}.acc = acc;
    log_print(result_file, '******%g\t%g @ round=%d, C=%g, lambda_L=%g, lambda_D1=%g, lambda_D2=%g, thr=%g\n', ap, acc, r, C, lambda_L, lambda_D1,lambda_D2, thr);
end
log_print(result_file, '<==========  END @ %s, C = %g, lambda_L = %g, lambda_D1 = %g, lambda_D2, thr = %g, beta = %g ============>\n', datestr(now), C, lambda_L, lambda_D1, lambda_D2, thr, beta);