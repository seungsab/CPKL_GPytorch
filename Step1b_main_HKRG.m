
%% INITIALIZATION
clc, clear all, close all, warning('off','all')

%% s t a r t the UQLab framework
addpath(genpath([pwd '\Utils']));
uqlab;

Case_IND = 1:3;

% 3:10
N_pts00 = 2:10;


% Start the UQLab framework
uqlab;

%%
fn_ld = 'Case_study2';
for ind0 = 1:size(N_pts00,2)
    clearvars -except N_pts00 ind0 fn_ld Case_IND
    load(fn_ld)
    
    N_pts = N_pts00(ind0);
    N_MC = 100;
    
    if N_pts+1<10
        load(['result_NM_0' num2str(N_pts+1)]);
    else
        load(['result_NM_' num2str(N_pts+1)]);
    end
    
    % Color
    color1 = [0 0 1];
    color2 = [1 0 0];
    % color2 = [217,95,2]/255;
    
    RMSE = [];
    
    for i_MC = 1:N_MC
        IND_success = 0;
        while ~IND_success
            IND = Ch_IND(i_MC,:);
            
            % Expensive points
            xe = xe0(IND); ye = mean(ye0(IND,:),2);
            
            % Cheap points
            xc = BOCDA.X'; yc = mean(BOCDA.Y(Case_IND,:)',2);
            
            disp([num2str(i_MC) ' // ' num2str(N_MC) ' (# HF: ' num2str(size(xe,1)) ' EA)']);
                        
            %% MF- GPmodel via Hierachical Kriging            
            % Create the low-fidelity surrogate
            KOptions_LF.Type = 'Metamodel';
            KOptions_LF.MetaType = 'Kriging';
            KOptions_LF.Trend.Type = 'ordinary'; % simple // ordinary
            
            KOptions_LF.ExpDesign.X = xc;
            KOptions_LF.ExpDesign.Y = yc;
            KOptions_LF.ExpDesign.Sampling = 'User';
            
            KOptions_LF.Corr.Family = 'Gaussian'; % Linear // Exponential // Gaussian // Matern-3_2 // Matern-5_2
            KOptions_LF.Regression.SigmaNSQ = 'auto'; % Noise on // off
            KOptions_LF.Optim.Method = 'BFGS';
            
            myKriging_LF = uq_createModel(KOptions_LF);
            
            % Create the high-fidelity surrogate
            KOptions_HF.Type = 'Metamodel';
            KOptions_HF.MetaType = 'Kriging';
            KOptions_HF.Trend.Type = 'ordinary'; % simple // ordinary
            
            KOptions_HF.ExpDesign.X = xe;
            KOptions_HF.ExpDesign.Y = ye;
            KOptions_HF.ExpDesign.Sampling = 'User';
            
            KOptions_HF.Corr.Family = 'Gaussian'; % Linear // Exponential // Gaussian // Matern-3_2 // Matern-5_2
            KOptions_HF.Regression.SigmaNSQ = 'auto'; % Noise on // off
            KOptions_HF.Optim.Method = 'BFGS';
            
            myKriging_HF = uq_createModel(KOptions_HF);
            
            % Create the hierarchical Kriging surrogate
            
            KOptions_Hier.Type = 'Metamodel';
            KOptions_Hier.MetaType = 'Kriging';
            KOptions_Hier.Trend.Type = 'ordinary'; % simple // ordinary
            
            KOptions_Hier.ExpDesign.X = xe;
            KOptions_Hier.ExpDesign.Y = ye;
            KOptions_Hier.ExpDesign.Sampling = 'User';
            
            KOptions_Hier.Corr.Family = 'Gaussian'; % Linear // Exponential // Gaussian // Matern-3_2 // Matern-5_2
            KOptions_Hier.Regression.SigmaNSQ = 'auto'; % Noise on // off
            KOptions_Hier.Optim.Method = 'BFGS'; % Noise on // off
            
            KOptions_Hier.Trend.CustomF = @(x) uq_evalModel(myKriging_LF, x);
            myKriging_Hier = uq_createModel(KOptions_Hier);
            
            YPRED_MF = uq_evalModel(myKriging_Hier, xTest);
            
            MF1.gpr_MF{i_MC,1} = myKriging_Hier;
            MF1.xTest(i_MC,:) = xTest; MF1.YPRED(i_MC,:) = YPRED_MF;
            disp('Done');
            
            %% COMPUTE Evaluation meaures
            [R2 RSM RAAE RMAE PBIAS] = Evaluate_performance(ye_Test, MF1.YPRED(i_MC,:)');
            MF1.ACCURACY(i_MC,:) = [R2 RSM RAAE RMAE PBIAS];
                        
%             Ch_IND = [Ch_IND; IND];
            IND_success = 1;
        end
    end
    
    if size(xe,1)<10
        save(['HKRG_result_NM_0' num2str(size(xe,1))]);
    else
        save(['HKRG_result_NM_' num2str(size(xe,1))]);
    end
    
    
    close all
    
    %% Plot
    [B(3,1) IX(3,1)] = min(MF1.ACCURACY(:,2)); [B(3,2) IX(3,2)] = max(MF1.ACCURACY(:,2));
    
%     PLOT_RESULT_BEST_WORST
end