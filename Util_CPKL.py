from copy import deepcopy
import torch
import gpytorch

import numpy as np
from matplotlib import pyplot as plt

from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import max_error as MAE
from matplotlib import cm

import varstool.sampling.lhs as lhs
import varstool.sampling.symlhs as symlhs


def initialize_hyperparameters(x, y):
    # Save mean/std of input for standardization
    y_mean, y_std = y.mean(), y.std()
    x_mean, x_std = x.mean(dim=0, keepdim=True), x.std(dim=0, keepdim=True)

    x_nor, y_nor, = deepcopy(x), deepcopy(y)

    # Standardize targets
    y_nor -= y_mean
    y_nor /= y_std

    # Standardize features
    x_nor -= x_mean
    x_nor /= x_std

    # Compute data patterns for initializing hyperparameters
    traindata_info = {}
    if x_nor.dim() == 1:  # 0-order tensor (Scaler)
        n_params = 1
    else:
        n_params = x.shape[1]

    traindata_info['n_param'] = n_params
    traindata_info['min_dist_x'] = np.abs(
        x_nor.reshape(-1, 1) @ x_nor.reshape(1, -1)).reshape(-1).min().item()
    traindata_info['med_dist_x'] = np.median(
        np.abs(x_nor.reshape(-1, 1) @ x_nor.reshape(1, -1)).reshape(-1)).item()
    traindata_info['max_dist_x'] = np.abs(
        x_nor.reshape(-1, 1) @ x_nor.reshape(1, -1)).reshape(-1).max().item()
    traindata_info['min_dist_y'] = np.abs(
        y_nor.reshape(-1, 1) @ y_nor.reshape(1, -1)).reshape(-1).min().item()
    traindata_info['med_dist_y'] = np.median(
        np.abs(y_nor.reshape(-1, 1) @ y_nor.reshape(1, -1)).reshape(-1)).item()
    traindata_info['max_dist_y'] = np.abs(
        y_nor.reshape(-1, 1) @ y_nor.reshape(1, -1)).reshape(-1).max().item()

    return traindata_info


def plot_result(model, test_x, test_y, plot_on=True, fn_sv=None):
    train_x = model.train_inputs[0]
    train_y = model.train_targets

    if model.options['standardize']:
        # Rescale prediction to original training data scale
        original_mean, original_std = model.x_mean.detach(
        ).numpy(), model.x_std.detach().numpy()
        train_x = train_x * original_std + original_mean
        original_mean, original_std = model.y_mean.detach(
        ).numpy(), model.y_std.detach().numpy()
        train_y = train_y*original_std + original_mean

    bic = model.calculate_aicbic()
    ypred_mu, ypred_var = model.predict(test_x)

    # Compute error metric    
    mae, mse = MAE(test_y.numpy(), ypred_mu), MSE(test_y.numpy(), ypred_mu)
    if plot_on:
        if model.train_inputs[0].shape[1] == 2:
            plt.figure(figsize=(5, 5), dpi = 200)
            plt.plot(ypred_mu, test_y.numpy(), 'o')
            xmin_, xmax_ = plt.gca().get_xlim()
            plt.plot([xmin_, xmax_], [xmin_, xmax_], 'r:', label = 'x = y')
            plt.xlabel('Prediction')
            plt.ylabel('Observation')

            if model.bic is not None:
                plt.title(
                    f'{model.kernel_type} // bic = {bic.item():.3f}, {mae:.3f} & {mse:.3f}', fontsize=12)

            if fn_sv:
                plt.savefig(fn_sv)
            plt.show()

        else: # 1D-data
            fig, axs = plt.subplots(1, 2, figsize=(8, 3))
            # Plot training data as black stars
            axs[0].plot(train_x.numpy(), train_y.numpy(), 'k*', label='Observed')
            axs[0].plot(test_x.numpy(), test_y.numpy(), 'c:',
                    label='Target fcnt', linewidth=2)
            
            # Plot predictive means as blue line
            axs[0].plot(test_x.numpy(), ypred_mu, 'r')

            # Shade between the lower and upper confidence bounds
            axs[0].fill_between(test_x.numpy(), ypred_mu - 3 * ypred_var ** 0.5,
                            ypred_mu + 3 * ypred_var ** 0.5, alpha=0.5, color='#e35f62')
            
            axs[0].set_xlabel('x')
            axs[0].set_ylabel('Strain ($\mu \epsilon$)')
            axs[0].grid(color='lightgray')
            axs[0].legend(fontsize=7, loc='upper left')

            axs[1].plot(ypred_mu, test_y.numpy(), 'o')
            xmin_, xmax_ = axs[1].get_xlim()
            axs[1].plot([xmin_, xmax_], [xmin_, xmax_], 'r:', label = 'x = y')
            axs[1].set_xlabel('Prediction')
            axs[1].set_ylabel('Observation')
                        
            if bic is not None:
                fig.suptitle(
                    f'{model.kernel_type} // bic = {bic.item():.3f}, {mae:.3f} & {mse:.3f}', fontsize=12)

            if fn_sv:
                plt.savefig(fn_sv)

            plt.show()

    return mae, mse


def generate_kernel_hyperparameter_for_multiple_restart(kernel_, kernel_type, multi_restart_option, rand_seed0):
    '''
        Generate candidates for kernel hyperparameters for optimization based on Multiple-start
    '''

    kernels, n_eval, n_param = [
    ], multi_restart_option['n_eval'], multi_restart_option['traindata_info']['n_param']

    if multi_restart_option['sampling'] == 'random':  # Random sampling
        for i in range(n_eval):
            kernel = deepcopy(kernel_)
            if kernel_type in ['rbf', 'rq', 'mat12', 'mat32', 'mat52']:
                kernel.raw_lengthscale = torch.nn.Parameter(torch.tensor(
                    [kernel.lengthscale_prior.rsample().item() for _ in range(n_param)]).view(1, n_param))

            elif kernel_type == 'lin':
                if n_param == 1:
                    kernel.raw_variance = torch.nn.Parameter(
                        torch.tensor(kernel.variance_prior.rsample().item()).view(1, n_param))
                else:
                    for i in range(n_param):
                        kernel.kernels[i].raw_variance = torch.nn.Parameter(
                            torch.tensor(kernel.kernels[i].variance_prior.rsample().item()).view(1, 1))

            elif kernel_type[:3] == 'cos':
                if n_param == 1:
                    kernel.raw_period_length = torch.nn.Parameter(
                        torch.tensor(kernel.period_length_prior.rsample().item()).view(1, n_param))
                else:
                    for i in range(n_param):
                        kernel.kernels[i].raw_period_length = torch.nn.Parameter(
                            torch.tensor(kernel.kernels[i].period_length_prior.rsample().item()).view(1, 1))

            else:  # 'per'
                kernel.raw_lengthscale = torch.nn.Parameter(torch.tensor(
                    [kernel.lengthscale_prior.rsample().item() for _ in range(n_param)]).view(1, n_param))
                kernel.raw_period_length = torch.nn.Parameter(torch.tensor(
                    [kernel.period_length_prior.rsample().item() for _ in range(n_param)]).view(1, n_param))

            kernels.append(kernel)

    else:

        # Generate initial samples in normalized space [0, 1]^P from LHS sampling
        if n_param == 1:  # 0-order tensor (Scaler)

            x_lhs = lhs(
                sp=n_eval, params=1, seed=rand_seed0,
                criterion='maximin', iterations=20)

            x_lhs1 = lhs(
                sp=n_eval, params=1, seed=rand_seed0 * 10,
                criterion='maximin', iterations=20)

        # tensor of having more than 1st order (vector, matrice, and so on)
        else:

            x_lhs = symlhs(
                sp=n_eval, params=n_param, seed=rand_seed0,
                criterion='maximin', iterations=20)

            x_lhs1 = symlhs(
                sp=n_eval, params=n_param, seed=rand_seed0 * 10,
                criterion='maximin', iterations=20)

        if kernel_type in ['rbf', 'rq', 'mat12', 'mat32', 'mat52']:
            ub, lb = multi_restart_option['traindata_info']['min_dist_x'], multi_restart_option['traindata_info']['max_dist_x']
            x_hyp = (ub - lb) * x_lhs + lb

            for i in range(x_hyp.shape[0]):
                kernel = deepcopy(kernel_)
                kernel.raw_lengthscale = torch.nn.Parameter(
                    torch.tensor([x_hyp[i, j] for j in range(x_hyp.shape[1])]).view(1, n_param))
                kernels.append(kernel)

        elif kernel_type == 'lin':
            ub, lb = multi_restart_option['traindata_info']['min_dist_y'], multi_restart_option['traindata_info']['max_dist_y']
            x_hyp = (ub - lb) * x_lhs + lb

            for i in range(x_hyp.shape[0]):
                kernel = deepcopy(kernel_)
                if n_param == 1:
                    kernel.raw_variance = torch.nn.Parameter(
                        torch.tensor(x_hyp[i, 0]).view(1, n_param))
                else:
                    for j in range(n_param):
                        kernel.kernels[j].raw_variance = torch.nn.Parameter(
                            torch.tensor(x_hyp[i, j]).view(1, 1))
                kernels.append(kernel)

        elif kernel_type[:3] == 'cos':
            ub, lb = multi_restart_option['traindata_info']['min_dist_x'], multi_restart_option['traindata_info']['max_dist_x']
            x_hyp = (ub - lb) * x_lhs + lb

            for i in range(x_hyp.shape[0]):
                kernel = deepcopy(kernel_)
                if n_param == 1:
                    kernel.raw_period_length = torch.nn.Parameter(
                        torch.tensor(x_hyp[i]).view(1, n_param))
                else:
                    for j in range(n_param):
                        kernel.kernels[j].raw_period_length = torch.nn.Parameter(
                            torch.tensor(x_hyp[i, j]).view(1, 1))
                kernels.append(kernel)
        else:
            ub, lb = multi_restart_option['traindata_info']['min_dist_x'], multi_restart_option['traindata_info']['max_dist_x']
            x_hyp = (ub - lb) * x_lhs + lb
            x_hyp1 = (ub - lb) * x_lhs1 + lb

            for i in range(x_hyp.shape[0]):
                kernel = deepcopy(kernel_)
                kernel.raw_lengthscale = torch.nn.Parameter(
                    torch.tensor([x_hyp[i, j] for j in range(x_hyp.shape[1])]).view(1, n_param))
                kernel.raw_period_length = torch.nn.Parameter(
                    torch.tensor([x_hyp1[i, j] for j in range(x_hyp1.shape[1])]).view(1, n_param))
                kernels.append(kernel)

    return kernels


def train_best_via_multiple_restart(train_x, train_y, kernels, kernel_type, likelihood_, options):
    '''
        Ref.) https://github.com/cornellius-gp/gpytorch/discussions/1782
    '''
    # Restart training to avoid getting stuck in local minima
    score, models = [], []
    for kernel_ in kernels:
        if 0:
            # Initialize likelihood and model
            model = ExactGPModel(deepcopy(train_x), deepcopy(
                train_y), kernel_, kernel_type, deepcopy(likelihood_), options).float()
            loss = model.optimize()
            best_loss = min(loss)

            score.append(best_loss)
            models.append(model)
        else:
            try:
                # Initialize likelihood and model
                model = ExactGPModel(deepcopy(train_x), deepcopy(
                    train_y), kernel_, kernel_type, deepcopy(likelihood_), options).float()
                loss = model.optimize()
                best_loss = min(loss)

                score.append(best_loss)
                models.append(model)

                # print(f"[{i+1}//{len(kernels)}] kernel type: {kernel_type}, bic: {model.bic}")

            except Exception as e:
                # print(e)

                with gpytorch.settings.cholesky_jitter(1e-1):
                    # Initialize likelihood and model
                    model = ExactGPModel(deepcopy(train_x), deepcopy(
                        train_y), kernel_, kernel_type, deepcopy(likelihood_), options).float()
                    loss = model.optimize()
                    best_loss = min(loss)

                    score.append(best_loss)
                    models.append(model)

            finally:
                continue
    if score:
        best_index = np.argmin(score, axis=0)
        return models[best_index]
    else:
        return models


def fixed_learned_Values_in_base_kernel(model):
    model_ = deepcopy(model)

    for param_name, param in model_.base_kernel.named_parameters():
        param.requires_grad = False

    return model_


def RUN_CPKL(
        train_x, train_y, likelihood, base_kernels,
        test_x=None, test_y=None,
        max_depth=5, multi_restart_option=None, rand_seed0=1234,
        training_iter=50, standardize=True, tolerance=1e-3,
        fix_value_learned=False, plot_intermediate_on=True
):

    if multi_restart_option == None:
        multi_restart_option['sampling'] = 'random'
        multi_restart_option['n_eval'] = 20

    search_on, i_depth = 1, 1
    options = {'training_iter': training_iter,
            'standardize': standardize, 'tolerance': tolerance}

    bic_old, best_models, best_bics, best_metrics = np.inf, [], [], []
    while search_on:
        bics, models = [], []

        # Run best-first search for CPKL
        if i_depth == 1:  # 1st depth => single base kernels
            for key in base_kernels.keys():
                kernel_type = key
                kernel = base_kernels[key]()
                kernels = generate_kernel_hyperparameter_for_multiple_restart(
                    kernel, key, multi_restart_option, rand_seed0)
                model = train_best_via_multiple_restart(
                    deepcopy(train_x), deepcopy(train_y), kernels,
                    kernel_type, deepcopy(likelihood), options)

                try:
                    # print(f"[{kernel_type}] best bic: {model.bic.item():.4f}", end='\n')
                    bics.append(model.bic.item())
                    models.append(model)
                except:
                    print(f"[{kernel_type}] failed ")

        else:  # more than 2nd depth => compositional kernels
            for key in base_kernels.keys():
                kernel = base_kernels[key]()
                kernels = generate_kernel_hyperparameter_for_multiple_restart(
                    kernel, key, multi_restart_option, rand_seed0)

                kernel_type = f"({model0.kernel_type}+{key})"
                new_CPKs = [deepcopy(model0.base_kernel) +
                            kernel_add for kernel_add in kernels]

                model = train_best_via_multiple_restart(
                    deepcopy(train_x), deepcopy(train_y), new_CPKs,
                    kernel_type, deepcopy(likelihood), options)

                try:
                    # print(f"[{kernel_type}] best bic: {model.bic.item():.4f}", end='\n')
                    bics.append(model.bic.item())
                    models.append(model)
                except:
                    print(f"[{kernel_type}] failed ")

            for key in base_kernels.keys():
                kernel = base_kernels[key]()
                kernels = generate_kernel_hyperparameter_for_multiple_restart(
                    kernel, key, multi_restart_option, rand_seed0)

                kernel_type = f"({model0.kernel_type}*{key})"
                new_CPKs = [deepcopy(model0.base_kernel)
                            * kernel_add for kernel_add in kernels]

                model = train_best_via_multiple_restart(
                    deepcopy(train_x), deepcopy(train_y), new_CPKs,
                    kernel_type, deepcopy(likelihood), options)

                try:
                    # print(f"[{kernel_type}] best bic: {model.bic.item():.4f}", end='\n')
                    bics.append(model.bic.item())
                    models.append(model)
                except:
                    print(f"[{kernel_type}] failed ")

        # Select best kernel from candidates
        best_index = np.argmin(bics, axis=0)
        best_model = models[best_index]

        best_models.append(best_model)
        best_bics.append(best_model.bic.item())
        bic_new = best_bics[-1]

        # Plot result
        mae, mse = plot_result(best_model, test_x, test_y,
                            plot_on=plot_intermediate_on)
        best_metrics.append([mae, mse])

        # Check whether loop for CPKL-search stops or not
        if (bic_old < bic_new) | (i_depth == max_depth):
            search_on = 0

            best_index = np.argmin(best_bics, axis=0)
            final_model = best_models[best_index]

            return final_model, best_models, best_metrics, best_index
        else:
            bic_old = bic_new
            model0 = deepcopy(best_model)
            # Fix values of hyper-parameters in prvious stage by removing "required_grad" in torch-backend
            if fix_value_learned:
                model0 = fixed_learned_Values_in_base_kernel(model0)
            i_depth += 1


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, kernel, kernel_type, likelihood, options):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        # self.mean_module = gpytorch.means.ZeroMean()
        self.base_kernel = kernel
        self.covar_module = gpytorch.kernels.ScaleKernel(self.base_kernel)
        self.kernel_type = kernel_type
        self.options = options

        if self.options['standardize']:
            # Save mean/std of input for standardization
            self.y_mean = train_y.mean()
            self.y_std = train_y.std()
            self.x_mean = train_x.mean(dim=0, keepdim=True)
            self.x_std = train_x.std(dim=0, keepdim=True)
            self.standardize_training_data()

    def standardize_training_data(self):
        # Standardize targets
        self.train_targets -= self.y_mean
        self.train_targets /= self.y_std
        # Standardize features
        train_x = self.train_inputs[0]
        train_x -= self.x_mean
        train_x /= self.x_std
        self.train_inputs = (train_x,)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def optimize(self, plot=False):
        model = self
        likelihood = self.likelihood
        X = self.train_inputs[0]
        y = self.train_targets

        model.train()
        likelihood.train()

        from LBFGS import LBFGS, FullBatchLBFGS

        optimizer = FullBatchLBFGS(model.parameters())
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        def closure():
            optimizer.zero_grad()
            output = model(X)
            loss = -mll(output, y)  # reach MLE through gradient descent
            return loss

        loss_trace = []
        i_iter, stop_flag = 0, 1

        with gpytorch.settings.cg_tolerance(0.01), gpytorch.settings.cg_tolerance(10000), gpytorch.settings.max_preconditioner_size(100):
            # Set gradients from previous iteration to 0
            loss = closure()
            loss.backward()

            while stop_flag:
                # perform step and update curvature
                options = {'closure': closure,
                           'current_loss': loss, 'max_ls': 10}
                loss, _, lr, _, F_eval, G_eval, _, _ = optimizer.step(options)
                loss_trace.append(loss.detach().numpy())
                loss_new = loss.item()
                if i_iter == 0:
                    loss_old = loss_new
                else:
                    if i_iter > self.options['training_iter']:
                        stop_flag = 0
                    elif np.abs((loss_old-loss_new)/loss_old) < self.options['tolerance']:
                        stop_flag = 0
                    else:
                        loss_old = loss_new

                i_iter += 1

            if plot:
                _, ax = plt.subplots(figsize=(8, 6))
                ax.set_xlabel("Training iteration")
                ax.set_ylabel("Marginal Log Likelihood Loss")
                ax.plot(loss_trace)

        self.loss_trace = loss_trace
        self.bic = model.calculate_aicbic()
        return loss_trace

    def predict(self, x):
        self.eval()
        self.likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            if isinstance(x, np.ndarray):
                x_pred = torch.from_numpy(x).type(torch.FloatTensor)
            elif torch.is_tensor(x):
                x_pred = x

            if self.options['standardize']:
                # Standardize prediction features
                x_pred = (x_pred - self.x_mean) / self.x_std

            try:
                prediction = self.likelihood(self(x_pred))
            except:
                with gpytorch.settings.cholesky_jitter(1e-1):
                    prediction = self.likelihood(self(x_pred))

            mean = prediction.mean.detach().numpy()
            var = prediction.variance.detach().numpy()

            if self.options['standardize']:
                # Rescale prediction to original training data scale
                original_mean = self.y_mean.detach().numpy()
                original_std = self.y_std.detach().numpy()
                mean = mean*original_std + original_mean
                var = var*original_std**2  # Variance is stationary and is only changed by a factor

            return mean, var

    def log_marginal_likelihood(self):
        # https://gist.github.com/bcolloran/7686b63a5544295f0382415e553194ea
        train_x = self.train_inputs[0].double()
        train_y = self.train_targets.double()

        L_gptorch = self.covar_module(
            train_x).cpu().detach().cholesky().numpy()
        L_inv = np.linalg.inv(L_gptorch)

        N = train_y.shape[0]
        sigma2 = self.likelihood.noise.item()

        K = L_gptorch @ L_gptorch.T + sigma2 * np.eye(N)
        K_inv = np.linalg.inv(K)

        sign, log_det_K = np.linalg.slogdet(K)

        return -0.5 * (
            train_y.T @ K_inv @ train_y + log_det_K + N*np.log(2 * np.pi))

    def calculate_aicbic(self):
        # N: # of samples
        # mll: margianl log-likelihood
        # p: # of parameters

        # output from model
        train_x = self.train_inputs[0]
        mll = self.log_marginal_likelihood()

        n = train_x.shape[0]
        k = sum(dict((p.data_ptr(), p.numel())
                for p in self.parameters()).values())

        self.bic = -2 * mll + k * np.log(n)

        return self.bic
