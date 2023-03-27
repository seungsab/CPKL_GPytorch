

%% INITIALIZATION
clc, clear all, close all, warning('off','all')

addpath(genpath([pwd '\Utils']));


% 4:10
N_pts0 = 2:10;

%% Import Trianing Samples
fn_ld = 'Case_study2';
Case_IND = 1:3;

%%
for ind0 = 1:size(N_pts0,2)
    clearvars -except N_pts0 ind0 fn_ld Case_IND
    load(fn_ld)
    
    global ModelInfo
    
    N_MC = 100;
    
    % Color
    color_type = [
        0.00 0.45 0.74
        0.85 0.33 0.10
        0.93 0.69 0.13
        0.49 0.18 0.56
        0.47 0.67 0.19
        0.30,0.75,0.93
        ];
    
    % 2 - 10
    N_pts = N_pts0(ind0);
    
    %% Import Trianing Samples
    Ne = size(GAUGE.X',1);
    
    % Calulate expensive observations
    xe0 = GAUGE.X'; ye0 = GAUGE.Y(Case_IND,:)';
    
    % Obtain Test data
    xTest = xe0; ye_Test = mean(ye0,2);
    
    mySeed=57; % an integer number
    rng(mySeed,'twister') %You can replace 'twister' with other generators
    
    N_pts = round(1:Ne/(N_pts+1):Ne);
    N_pts = [N_pts Ne+1];
    
    Ch_IND = [];    
    RMSE = [];
    
    for i_MC = 1:N_MC
        IND_success = 0;
        while ~IND_success
            IND = [];
            for i = 1 : size(N_pts,2)-1
                IND = [IND randi([N_pts(i) N_pts(i+1)-1],1,1)];
            end
            
            % Expensive points
            xe = xe0(IND); ye = mean(ye0(IND,:),2);
            
            % Cheap points
            xc = BOCDA.X'; yc = mean(BOCDA.Y(Case_IND,:)',2);
            
            disp([num2str(i_MC) ' // ' num2str(N_MC) ' (# HF: ' num2str(size(xe,1)) ' EA)']);
            
            %% HF model via SE kernel
            fprintf('Modeling GP-HF (SE)...');
            Best = GP_SINGLE_KERNEL_STRUCTURE(xe,ye,1); % SE Kernel
            BIC_HF = Best.BIC; hyp_HF = Best.hyp; covfunc_HF = Best.covfunc;
            [YPRED_HF, YSD_HF] = gp(hyp_HF, @infGaussLik, [], covfunc_HF, @likGauss, xe, ye, xTest); % Predict via GP-HF
            
            HF.BIC(i_MC,:) = BIC_HF;
            HF.xe(i_MC,:) = xe;  HF.ye(i_MC,:) = ye;
            HF.gpr_MF{i_MC,1} = Best;  HF.xTest(i_MC,:) = xTest;
            HF.YPRED(i_MC,:) = YPRED_HF; HF.YSD_HF(i_MC,:) = YSD_HF;
            disp('Done');
            %% LF-GP model
            fprintf('Modeling GP-LF (SE)...');
            Best = GP_SINGLE_KERNEL_STRUCTURE(xc,yc,1); % SE Kernel
            BIC_LF = Best.BIC; hyp_LF = Best.hyp; covfunc_LF = Best.covfunc;
            [YPRED_LF, YSD_LF] = gp(hyp_LF, @infGaussLik, [], covfunc_LF, @likGauss, xc, yc, xTest); % Predict test data via GP-LF
            [y_hat_LF, y_hat_SD_LF] = gp(hyp_LF, @infGaussLik, [], covfunc_LF, @likGauss, xc, yc, xe); % Predict HF-data via GP-LF
            
            LF.BIC(i_MC,:) = BIC_LF;
            LF.xc(i_MC,:) = xc;  LF.yc(i_MC,:) = yc;
            LF.gpr_MF{i_MC,1} = Best;  LF.xTest(i_MC,:) = xTest;
            LF.YPRED(i_MC,:) = YPRED_LF; LF.YSD_LF(i_MC,:) = YSD_LF;
            disp('Done');
            
            %% MF- GPmodel via linear mapping
            fprintf('Modeling GP-MF (Linear)...');
            obj = @(rho,y_hat_LF,ye)sum((rho * y_hat_LF- ye).^2);
            fun = @(rho) obj(rho,y_hat_LF,ye);
            rho = fminsearch(fun,1);
            d = ye - rho * y_hat_LF;
            
            Best_d = GP_SINGLE_KERNEL_STRUCTURE(xe,d,1); % SE Kernel
            BIC_d = Best_d.BIC; hyp_d = Best_d.hyp; covfunc_d = Best_d.covfunc;
            [y_dhat_MF1, YSD_dhat_MF1] = gp(hyp_d, @infGaussLik, [], covfunc_d, @likGauss, xe, d, xTest); % Predict test data
            
            YPRED_MF = rho * YPRED_LF + y_dhat_MF1;
            
            MF1.BIC(i_MC,:) = BIC_d;
            MF1.gpr_RHO(i_MC,:) = rho; MF1.gpr_MF{i_MC,1} = Best_d;
            MF1.xTest(i_MC,:) = xTest; MF1.YPRED(i_MC,:) = YPRED_MF;
            disp('Done');
            
            %% MF- GPmodel via Kennedy-O'hagan method
            fprintf('Modeling GP-MF (KO)...');
            % Define Class for Multi-fidelity Gaussian Process (MFGP) Regression
            k = size(xe,2); % Input dimension
            ModelInfo.Xe = xe; ModelInfo.ye = ye; % HF data
            ModelInfo.Xc = xc; ModelInfo.yc = yc; % LF data
            
            % Estimate hyper-parameters of LF model
            [Pars, NLML] = ga_MF(@likelihoodc_MF,k+1,[],[],[],[],[-3 -10],[3 3]);
            ModelInfo.Thetac=Pars(1:k);
            ModelInfo.lambdac=Pars(end);
            % Optimise difference model paramters
            [Pars, NLML] = ga_MF(@likelihoodd_MF,k+2,[],[],[],[],[-3 -5 -10],[3 5 3]);
            ModelInfo.Thetad=Pars(1:k);
            ModelInfo.rho=Pars(k+1);
            ModelInfo.lambdae=Pars(end);
            
            % Construct covariance matrix
            buildcokriging_MF
            
            % PREDICT
            for i00 = 1:size(xTest,1)
                ModelInfo.Option='Pred';
                YPRED_MF(i00)=cokrigingpredictor(xTest(i00));
                
                ModelInfo.Option='RMSE';
                YSD_MF(i00)=cokrigingpredictor(xTest(i00));
            end
            
            MF2.xe(i_MC,:) = xe;  MF2.ye(i_MC,:) = ye;
            MF2.xc(i_MC,:) = xc;  MF2.yc(i_MC,:) = yc;
            MF2.xTest(i_MC,:) = xTest;
            MF2.YPRED(i_MC,:) = YPRED_MF; MF2.YSD_LF(i_MC,:) = YSD_MF;
            disp('Done');
            
            
            %% Proposed method #1: SE
            fprintf('Modeling GP-MF (GP-mapping(SE))...');
            % Step #1: LF-GP model
            YPRED_LF; % Predict test data via GP-LF
            ye_hat_LF = y_hat_LF; % Predict HF-data via GP-LF
            
            % Step #2: GP Mapping from LF to HF
            XX = [xe ye_hat_LF];
            Best_rho = GP_SINGLE_KERNEL_STRUCTURE(XX,ye,1); % SE Kernel
            BIC_rho = Best_rho.BIC; hyp_rho = Best_rho.hyp; covfunc_rho = Best_rho.covfunc;
            [y_hat_RHO, YSD_RHO] = gp(hyp_rho, @infGaussLik, [], covfunc_rho, @likGauss, XX, ye, XX); % Predict HF-data via GP-GPmapping(SE)
            [ytest_hat_RHO, YtestSD_RHO] = gp(hyp_rho, @infGaussLik, [], covfunc_rho, @likGauss, XX, ye, [xTest YPRED_LF]); % Predict testdata via GP-GPmapping(SE)
            
            % Step #3: GP for model discrepancy
            d = ye - y_hat_RHO;
            Best_d = GP_SINGLE_KERNEL_STRUCTURE(xe,d,1); % SE Kernel
            BIC_d = Best_d.BIC; hyp_d = Best_d.hyp; covfunc_d = Best_d.covfunc;
            [y_d, YSD_hat2] = gp(hyp_d, @infGaussLik, [], covfunc_d, @likGauss, xe, d, xTest);
            
            YPRED_MF = ytest_hat_RHO + y_d;
            
            MF3.xe(i_MC,:) = xe;  MF3.ye(i_MC,:) = ye;
            MF3.xc(i_MC,:) = xe;  MF3.yc(i_MC,:) = yc;
            MF3.gpr_RHO{i_MC,1} = Best_rho; MF3.gpr_MF{i_MC,1} = Best_d;
            MF3.xTest(i_MC,:) = xTest; MF3.ye_Test(i_MC,:) = ye_Test;
            MF3.y_hat_RHO(i_MC,:) = y_hat_RHO; MF3.y_d(i_MC,:) = y_d;
            MF3.YPRED(i_MC,:) = YPRED_MF;
            disp('Done');
            
            if 0
                %% Proposed method #2: CPK
                fprintf('Modeling GP-MF (GP-mapping(CPK))...');
                % Step #1: LF-GP model
                [Best_LF, Final, BIC_hist] = GP_Structure_Discovery_CKL(xc,yc,10);
                BIC_LF = Best_LF.BIC; hyp_LF = Best_LF.hyp; covfunc_LF = Best_LF.covfunc;
                [y_hat_LF, y_hatSD_LF] = gp(hyp_LF, @infGaussLik, [], covfunc_LF, @likGauss, xc, yc, xe); % PREDICT HF-data via GP-LF(CPK)
                [y_hat1, YSD_hat1] = gp(hyp_LF, @infGaussLik, [], covfunc_LF, @likGauss, xc, yc, xTest);  % PREDICT testdata via GP-LF(CPK)
                
                % Step #2: GP Mapping from LF to HF
                XX = [xe y_hat_LF];
                [Best_rho, Final, BIC_hist] = GP_Structure_Discovery_CKL(XX,ye,10);
                BIC_rho = Best_rho.BIC; hyp_rho = Best_rho.hyp; covfunc_rho = Best_rho.covfunc;
                [y_hat_RHO, YSD_RHO] = gp(hyp_rho, @infGaussLik, [], covfunc_rho, @likGauss, XX, ye, XX); % PREDICT HF-data via GP-GPmapping(CPK)
                [ytest_hat_RHO, YtestSD_RHO] = gp(hyp_rho, @infGaussLik, [], covfunc_rho, @likGauss, XX, ye, [xTest y_hat1]); % PREDICT testdata via GP-GPmapping(CPK)
                
                % Step #3: GP for model discrepancy
                d = ye - y_hat_RHO;
                [Best_d, Final, BIC_hist] = GP_Structure_Discovery_CKL(xe,d,10);
                BIC_d = Best_d.BIC; hyp_d = Best_d.hyp; covfunc_d = Best_d.covfunc;
                [y_d, YSD_hat2] = gp(hyp_d, @infGaussLik, [], covfunc_d, @likGauss, xe, d, xTest);
                
                
                YPRED_MF = ytest_hat_RHO + y_d;
                
                MF4.xe(i_MC,:) = xe;  MF4.ye(i_MC,:) = ye;
                MF4.xc(i_MC,:) = xe;  MF4.yc(i_MC,:) = yc;
                MF4.gpr_LF{i_MC,1} = Best_LF; MF4.gpr_RHO{i_MC,1} = Best_rho; MF4.gpr_MF{i_MC,1} = Best_d;
                MF4.xTest(i_MC,:) = xTest; MF4.ye_Test(i_MC,:) = ye_Test;
                MF4.y_hat_RHO(i_MC,:) = y_hat_RHO; MF4.y_d(i_MC,:) = y_d;
                MF4.YPRED(i_MC,:) = YPRED_MF;
                disp('Done');
            end
            %% COMPUTE Evaluation meaures
            [R2 RSM RAAE RMAE PBIAS] = Evaluate_performance(ye_Test, MF1.YPRED(i_MC,:)');
            MF1.ACCURACY(i_MC,:) = [R2 RSM RAAE RMAE PBIAS];
            
            [R2 RSM RAAE RMAE PBIAS] = Evaluate_performance(ye_Test, MF2.YPRED(i_MC,:)');
            MF2.ACCURACY(i_MC,:) = [R2 RSM RAAE RMAE PBIAS];
            
            [R2 RSM RAAE RMAE PBIAS] = Evaluate_performance(ye_Test, MF3.YPRED(i_MC,:)');
            MF3.ACCURACY(i_MC,:) = [R2 RSM RAAE RMAE PBIAS];
            
%             [R2 RSM RAAE RMAE PBIAS] = Evaluate_performance(ye_Test, MF4.YPRED(i_MC,:)');
%             MF4.ACCURACY(i_MC,:) = [R2 RSM RAAE RMAE PBIAS];
            
            [R2 RSM RAAE RMAE PBIAS] = Evaluate_performance(ye_Test, YPRED_HF);
            HF.ACCURACY(i_MC,:) = [R2 RSM RAAE RMAE PBIAS];
            
            [R2 RSM RAAE RMAE PBIAS] = Evaluate_performance(ye_Test, YPRED_LF);
            LF.ACCURACY(i_MC,:) = [R2 RSM RAAE RMAE PBIAS];
            
            Ch_IND = [Ch_IND; IND];
            IND_success = 1;
        end
    end
    
    if size(xe,1)<10
        save(['result_NM_0' num2str(size(xe,1))]);
    else
        save(['result_NM_' num2str(size(xe,1))]);
    end
    
    
    close all
    
    %% Plot
    % Best  IX(1,1) // IX(2,1)
    % Worst IX(2,1) // IX(2,2)
    [B(1,1) IX(1,1)] = min(LF.ACCURACY(:,2)); [B(1,2) IX(1,2)] = max(LF.ACCURACY(:,2));
    [B(2,1) IX(2,1)] = min(HF.ACCURACY(:,2)); [B(2,2) IX(2,2)] = max(HF.ACCURACY(:,2));
    [B(3,1) IX(3,1)] = min(MF2.ACCURACY(:,2)); [B(3,2) IX(3,2)] = max(MF2.ACCURACY(:,2));
    [B(4,1) IX(4,1)] = min(MF3.ACCURACY(:,2)); [B(4,2) IX(4,2)] = max(MF3.ACCURACY(:,2));
%     [B(5,1) IX(5,1)] = min(MF3.ACCURACY(:,2)); [B(5,2) IX(5,2)] = max(MF3.ACCURACY(:,2));
%     [B(6,1) IX(6,1)] = min(MF4.ACCURACY(:,2)); [B(6,2) IX(6,2)] = max(MF4.ACCURACY(:,2));
    
    PLOT_RESULT_BEST_WORST
end

