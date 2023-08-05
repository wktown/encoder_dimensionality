import logging
from typing import List, Dict

import torch
import numpy as np
import xarray as xr
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold, StratifiedShuffleSplit
from brainscore.utils.batched_regression import LinearRegressionBatched
from utils.time_logging import timer
from computations.matrices import (
    torch_batch_pairwise_pearsonr,
    torch_pairwise_mse,
)

logging.basicConfig(level=logging.INFO)


# default paths and parameters
MAX_EPOCHS = 1000
# MAX_EPOCHS = 1
MAX_EPOCHS_WITHOUT_ES = 100
TOL = 0.01
NUM_EPOCH_TOL = 24
COV_BATCH = 15000


class RegressionBatched:
    def __init__(
        self,
        device: str,
        batch_size: int = 250,
        max_epochs: int = MAX_EPOCHS,
        max_epochs_noes: int = MAX_EPOCHS_WITHOUT_ES,
        random_seed: int = 11,
    ):
        self._regression = LinearRegressionBatched(device=device)
        self._batch_size = batch_size
        self._max_epochs = max_epochs
        self._max_epochs_noes = max_epochs_noes
        self._random_seed = random_seed
        self._fitted = False

    # old function for concatenating all predictor models
    @timer(event="regression fitting", threshold=5.0)
    def fit_large(
        self,
        source: xr.DataArray,
        target: np.ndarray,
        chunk_idx: List[int],
        chunks_size: List[int],
    ) -> None:
        rng = np.random.default_rng(self._random_seed)
        for c in chunk_idx:
            temp = np.sum(chunks_size[:c], dtype="int")
            stim_idx = slice(temp, temp + chunks_size[c])
            sub_source, sub_target = (
                source.isel(stimulus_path=stim_idx).values,
                target[stim_idx],
            )
            sub_idx = np.arange(chunks_size[c])
            for _ in range(self._max_epochs_noes):
                rng.shuffle(sub_idx)
                for j in range(0, chunks_size[c], self._batch_size):
                    self._regression.fit_partial(
                        sub_source[sub_idx[j : j + self._batch_size]],
                        sub_target[sub_idx[j : j + self._batch_size]],
                    )
        self._fitted = True

    @timer(event="regression fitting", threshold=60.0)
    def fit(
        self,
        source: np.ndarray,
        target: np.ndarray,
        chunk_idx: List[int],
        chunks_size: List[int],
    ) -> None:
        rng = np.random.default_rng(self._random_seed)
        stim_idx = chunks_full_index(chunks_size, chunk_idx)
        sub_idx = np.arange(len(stim_idx))
        sub_source, sub_target = source[stim_idx], target[stim_idx]
        losses = []
        for e in range(self._max_epochs):
            rng.shuffle(sub_idx)
            within = []
            for j in range(0, len(stim_idx), self._batch_size):
                within.append(
                    self._regression.fit_partial(
                        sub_source[sub_idx[j : j + self._batch_size]],
                        sub_target[sub_idx[j : j + self._batch_size]],
                    )
                )
            if len(losses) >= NUM_EPOCH_TOL:
                if np.mean(within) / np.mean(losses[-NUM_EPOCH_TOL:]) > 1 - TOL:
                    # logging.info(f"Early stop at epoch {e}")
                    break
            losses.append(np.mean(within))
        self._fitted = True

    def predict(
        self, source, chunk_idx: List[int], chunks_size: List[int]
    ) -> np.ndarray:
        yhats = []
        for c in chunk_idx:
            temp = np.sum(chunks_size[:c], dtype="int")
            stim_idx = slice(temp, temp + chunks_size[c])
            if isinstance(source, xr.DataArray):
                sub_source = source.isel(stimulus_path=stim_idx).values
            else:
                sub_source = source[stim_idx]
            for j in range(0, chunks_size[c], self._batch_size):
                yhats.append(
                    self._regression.predict(sub_source[j : j + self._batch_size])
                )
        return np.concatenate(yhats, axis=0)


class CrossRegressionEvaluation:
    def __init__(
        self,
        device: str = None,
        regression: str = "ridge",
        metric: List[str] = ["r", "r2", "mse"],
        num_cv: int = 10,
        random_seed: int = 11,
        regularization: int = 1,
        n_components: int = 25,
        scale: bool = False,
        shuffle: bool = True,
    ):
        self._num_cv = num_cv
        self._random_seed = random_seed
        if device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = device
        if regression == "pls":
            if self._device == "cuda":
                self.regression = PLSRegressionPytorch
                self.regression_config = {
                    "n_components": n_components,
                    "scale": scale,
                    "device": self._device,
                }
            else:
                self.regression = PLSRegression
                self.regression_config = {"n_components": n_components, "scale": scale}
        elif regression == "ridge":
            self.regression = RidgeRegressionPytorch
            self.regression_config = {
                "regularization": regularization,
                "device": self._device,
            }
        elif regression == "ols":
            self.regression = LinearRegressionPytorch
            self.regression_config = {"device": self._device, "scale": scale}
            # self.regression = LinearRegression
            # self.regression_config = {}
        self._regression = regression
        if isinstance(metric, str):
            metric = [metric]
        self._metric = metric
        self._shuffle = shuffle

    def __call__(
        self, X: np.ndarray, y: np.ndarray, stratification: List[str] = None
    ) -> Dict[str, List[np.ndarray]]:
        if stratification is None:
            seed = self._random_seed if self._shuffle else None
            split = KFold(n_splits=self._num_cv, shuffle=self._shuffle, random_state=seed)
            split_kwargs = {}
        else:
            # for majaj-hong data (not used anymore)
            split = StratifiedShuffleSplit(
                n_splits=self._num_cv, train_size=0.9, random_state=self._random_seed
            )
            split_kwargs = {"y": stratification}
        scores = {}
        for m in self._metric:
            scores[m] = []
        assert X.shape[0] == y.shape[0]
        for train_index, test_index in split.split(X, **split_kwargs):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            model = self.regression(**self.regression_config)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            rtemp = None
            for m in self._metric:
                if m == "r":
                    rtemp = torch_batch_pairwise_pearsonr(
                        X=torch.Tensor(y_test.T), Y=torch.Tensor(y_pred.T)
                    )
                    assert len(rtemp) == y_test.shape[1]
                    scores[m].append(rtemp)
                # hard code warning
                elif m == "r2":
                    if rtemp is not None:
                        scores[m].append(rtemp**2)
                elif m == "mse":
                    temp = torch_pairwise_mse(
                        Y=torch.Tensor(y_test.T), Ypred=torch.Tensor(y_pred.T)
                    )
                    scores[m].append(temp)
        return scores


## the rest are adapted from sklearn or brain-score
def _get_first_singular_vectors_power_method(x, y, max_iter=500, tol=1e-06):
    eps = torch.finfo(x.dtype).eps
    y_score = next(col for col in y.T if torch.any(torch.abs(col) > eps))
    x_weights_old = 100
    for i in range(max_iter):
        x_weights = (x.T @ y_score) / (y_score @ y_score)
        x_weights /= torch.sqrt(x_weights @ x_weights) + eps
        x_score = x @ x_weights
        y_weights = (y.T @ x_score) / (x_score.T @ x_score)
        y_score = (y @ y_weights) / ((y_weights @ y_weights) + eps)
        x_weights_diff = x_weights - x_weights_old
        if (x_weights_diff @ x_weights_diff) < tol or y.shape[1] == 1:
            break
        x_weights_old = x_weights
    n_iter = i + 1
    return x_weights, y_weights, n_iter


def _svd_flip_1d(u, v):
    biggest_abs_val_idx = torch.argmax(torch.abs(u))
    sign = torch.sign(u[biggest_abs_val_idx])
    u *= sign
    v *= sign


class LinearRegressionPytorch:
    def __init__(
        self,
        device: str,
        fit_intercept: bool = True,
        scale: bool = False,
        remove_uniforms: bool = True
    ) -> None:
        self.fit_intercept = fit_intercept
        self.device_ = device
        self.n_features_in_ = None
        self.coef_ = None
        self.intercept_ = None
        self._residues = None
        self.rank_ = None
        self.singular_ = None
        self.alives_ = None
        self.scale = scale
        self._remove_uniforms = remove_uniforms

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        x = torch.from_numpy(x).float().to(self.device_)
        y = torch.from_numpy(y).float().to(self.device_)
        n_samples_, self.n_features_in_ = x.shape

        if self._remove_uniforms:
            x = x.T
            self.alives_ = [i for i, f in enumerate(x) if len(torch.unique(f)) > 1]
            x = x[self.alives_].T

        if self.scale:
            self.x_std, self.y_std = x.std(dim=0, unbiased=False), y.std(
                dim=0, unbiased=False
            )
            self.x_std[self.x_std == 0.0] = 1.0
            self.y_std[self.y_std == 0.0] = 1.0
            x /= self.x_std
            y /= self.y_std
        else:
            self.x_std, self.y_std = torch.ones(x.shape[1]), torch.ones(y.shape[1])

        if self.fit_intercept:
            x = torch.cat(
                [x, torch.ones(n_samples_, device=self.device_).unsqueeze(1)], dim=1
            )

        self.coef_, self._residues, self.rank_, self.singular_ = torch.linalg.lstsq(
            x, y
        )

        if self.fit_intercept:
            self.intercept_ = self.coef_[-1, :]
            self.coef_ = self.coef_[:-1, :]
        else:
            self.intercept_ = torch.zeros(self.n_features_in_)   
        self.coef_ *= self.y_std.to(self.device_)
        
    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("model has not been fit")
        else:
            x = torch.from_numpy(x).float().to(self.device_)
            if self._remove_uniforms:
                x = x.T
                x = x[self.alives_].T
            x /= self.x_std.to(self.device_)
            x = torch.matmul(x, self.coef_)
            x += self.intercept_
            return x.cpu().numpy()


class RidgeRegressionPytorch:
    def __init__(
        self,
        device: str,
        regularization: float = 1,
        fit_intercept: bool = True,
    ):
        self.regularization = regularization
        self.fit_intercept = fit_intercept
        self.device_ = device

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        x = torch.from_numpy(x).float().to(self.device_)
        y = torch.from_numpy(y).float().to(self.device_)
        assert x.shape[0] == y.shape[0], "number of X and y rows don't match"
        if self.fit_intercept:
            x = torch.cat([torch.ones(x.shape[0], 1, device=self.device_), x], dim=1)
        lhs = x.T @ x
        rhs = x.T @ y
        if self.regularization == 0:
            self.w, _ = torch.lstsq(rhs, lhs)
        else:
            ridge = self.regularization * torch.eye(lhs.shape[0], device=self.device_)
            self.w = torch.linalg.lstsq(lhs + ridge, rhs)[0]

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(x).float().to(self.device_)
        if self.fit_intercept:
            x = torch.cat([torch.ones(x.shape[0], 1, device=self.device_), x], dim=1)
        return (x @ self.w).cpu().numpy()


class PLSRegressionPytorch:
    def __init__(
        self,
        device,
        n_components: int = 25,
        scale: bool = True,
        max_iter: int = 500,
        tol=1e-06,
    ):
        self.n_components = n_components
        self.scale = scale
        self.max_iter = max_iter
        self.tol = tol
        self.device_ = device

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        x, y = torch.from_numpy(x).float().to(self.device_), torch.from_numpy(
            y
        ).float().to(self.device_)
        n, p = x.shape
        q = y.shape[1]
        self.x_mean, self.y_mean = x.mean(dim=0), y.mean(dim=0)
        x -= self.x_mean
        y -= self.y_mean

        if self.scale:
            self.x_std, self.y_std = x.std(dim=0, unbiased=False), y.std(
                dim=0, unbiased=False
            )
            self.x_std[self.x_std == 0.0] = 1.0
            self.y_std[self.y_std == 0.0] = 1.0
            x /= self.x_std
            y /= self.y_std
        else:
            self.x_std, self.y_std = torch.ones(p), torch.ones(q)

        self.x_weights_ = torch.zeros((p, self.n_components))  # U
        self.y_weights_ = torch.zeros((q, self.n_components))  # V
        self._x_scores = torch.zeros((n, self.n_components))  # Xi
        self._y_scores = torch.zeros((n, self.n_components))  # Omega
        self.x_loadings_ = torch.zeros((p, self.n_components))  # Gamma
        self.y_loadings_ = torch.zeros((q, self.n_components))  # Delta
        self.n_iter_ = []
        y_eps = torch.finfo(y.dtype).eps

        for k in range(self.n_components):
            y[:, torch.all(torch.abs(y) < 10 * y_eps, axis=0)] = 0.0

            try:
                (
                    x_weights,
                    y_weights,
                    n_iter_,
                ) = _get_first_singular_vectors_power_method(
                    x, y, max_iter=self.max_iter, tol=self.tol
                )
            except:
                logging.info(f"Y residual is constant at iteration {k}")
                self.x_weights_ = self.x_weights_[:, : k - 1]
                self.y_weights_ = self.y_weights_[:, : k - 1]
                self._x_scores = self._x_scores[:, : k - 1]
                self._y_scores = self._y_scores[:, : k - 1]
                self.x_loadings_ = self.x_loadings_[:, : k - 1]
                self.y_loadings_ = self.y_loadings_[:, : k - 1]
                break
            self.n_iter_.append(n_iter_)
            _svd_flip_1d(x_weights, y_weights)
            x_scores = x @ x_weights
            y_scores = (y @ y_weights) / (y_weights @ y_weights)

            # Deflation: subtract rank-one approx to obtain Xk+1 and Yk+1
            x_loadings = (x_scores @ x) / (x_scores @ x_scores)
            x -= torch.outer(x_scores, x_loadings)
            y_loadings = (x_scores @ y) / (x_scores @ x_scores)
            y -= torch.outer(x_scores, y_loadings)

            self.x_weights_[:, k] = x_weights
            self.y_weights_[:, k] = y_weights
            self._x_scores[:, k] = x_scores
            self._y_scores[:, k] = y_scores
            self.x_loadings_[:, k] = x_loadings
            self.y_loadings_[:, k] = y_loadings

        self.x_rotations_ = self.x_weights_ @ torch.linalg.pinv(
            self.x_loadings_.T @ self.x_weights_
        )
        self.y_rotations_ = self.y_weights_ @ torch.linalg.pinv(
            self.y_loadings_.T @ self.y_weights_
        )
        self.coef_ = (self.x_rotations_ @ self.y_loadings_.T).to(self.device_)
        self.coef_ = (self.coef_ * self.y_std.to(self.device_)).T
        self.intercept_ = self.y_mean
        self._n_features_out = self.x_rotations_.shape[1]

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(x).float().to(self.device_)
        x = (x - self.x_mean.to(self.device_)) / self.x_std.to(self.device_)
        return (
            ((x @ self.coef_.to(self.device_).T) + self.intercept_.to(self.device_))
            .cpu()
            .numpy()
        )


def chunks_full_index(chunks, chunk_idx) -> np.ndarray:
    chunk_idx = np.sort(chunk_idx)
    idx = np.concatenate(
        [
            np.arange(
                np.sum(chunks[:c], dtype="int"),
                np.sum(chunks[:c], dtype="int") + chunks[c],
            )
            for c in chunk_idx
        ]
    )
    return idx
