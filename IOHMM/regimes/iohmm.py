from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp


EPS = 1e-12


def _add_intercept(X: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(X)), X])


def _gaussian_logpdf(y: np.ndarray, mu: np.ndarray, sigma2: float) -> np.ndarray:
    sigma2 = max(float(sigma2), 1e-8)
    return -0.5 * (np.log(2.0 * np.pi * sigma2) + ((y - mu) ** 2) / sigma2)


def _standardize_fit(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.nanmean(X, axis=0)
    std = np.nanstd(X, axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    Xs = (X - mean) / std
    return Xs, mean, std


def _standardize_apply(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    std = np.where(std < 1e-12, 1.0, std)
    return (X - mean) / std


@dataclass
class GaussianEmissionState:
    beta: np.ndarray
    sigma2: float


class GaussianEmissionModel:
    def __init__(self, n_states: int, n_features: int, ridge: float = 1e-4, sigma2_floor: float = 1e-4):
        self.n_states = n_states
        self.n_features = n_features
        self.ridge = float(ridge)
        self.sigma2_floor = float(sigma2_floor)
        self.states: List[GaussianEmissionState] = []

    def initialize_from_quantiles(self, X: np.ndarray, y: np.ndarray) -> None:
        X1 = _add_intercept(X)
        qs = np.quantile(y, np.linspace(0.0, 1.0, self.n_states + 1))
        self.states = []

        for k in range(self.n_states):
            if k == self.n_states - 1:
                mask = (y >= qs[k]) & (y <= qs[k + 1])
            else:
                mask = (y >= qs[k]) & (y < qs[k + 1])

            if np.sum(mask) < max(5, X1.shape[1]):
                beta = np.zeros(X1.shape[1], dtype=float)
                beta[0] = float(np.mean(y))
                sigma2 = max(float(np.var(y)), self.sigma2_floor)
            else:
                beta = self._weighted_ridge_fit(X1, y, mask.astype(float))
                mu = X1 @ beta
                resid = y[mask] - mu[mask]
                sigma2 = max(float(np.mean(resid ** 2)), self.sigma2_floor)

            self.states.append(GaussianEmissionState(beta=beta, sigma2=sigma2))

    def _weighted_ridge_fit(self, X1: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        w = np.maximum(w, 0.0)
        W = np.sqrt(w)[:, None]
        Xw = X1 * W
        yw = y * W[:, 0]

        reg = self.ridge * np.eye(X1.shape[1], dtype=float)
        reg[0, 0] = 0.0
        A = Xw.T @ Xw + reg
        b = Xw.T @ yw
        return np.linalg.solve(A, b)

    def fit(self, X: np.ndarray, y: np.ndarray, gamma: np.ndarray) -> None:
        X1 = _add_intercept(X)
        if not self.states:
            self.initialize_from_quantiles(X, y)

        new_states: List[GaussianEmissionState] = []
        for k in range(self.n_states):
            w = gamma[:, k]
            beta = self._weighted_ridge_fit(X1, y, w)
            mu = X1 @ beta
            sigma2 = np.sum(w * (y - mu) ** 2) / max(np.sum(w), EPS)
            sigma2 = max(float(sigma2), self.sigma2_floor)
            new_states.append(GaussianEmissionState(beta=beta, sigma2=sigma2))

        self.states = new_states

    def means(self, X: np.ndarray) -> np.ndarray:
        X1 = _add_intercept(X)
        return np.column_stack([X1 @ st.beta for st in self.states])

    def log_likelihood_matrix(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        mus = self.means(X)
        ll = np.zeros((len(y), self.n_states), dtype=float)
        for k, st in enumerate(self.states):
            ll[:, k] = _gaussian_logpdf(y, mus[:, k], st.sigma2)
        return ll


class SoftmaxTransitionModel:
    def __init__(self, n_states: int, n_features: int, l2: float = 1e-3):
        self.n_states = n_states
        self.n_features = n_features
        self.l2 = float(l2)
        self.W = np.zeros((n_states, n_states, n_features + 1), dtype=float)

    def _project_reference(self) -> None:
        self.W[:, self.n_states - 1, :] = 0.0

    def _row_log_probs(self, X: np.ndarray, prev_state: int) -> np.ndarray:
        X1 = _add_intercept(X)
        logits = X1 @ self.W[prev_state].T
        logits = logits - logits.max(axis=1, keepdims=True)
        return logits - logsumexp(logits, axis=1, keepdims=True)

    def log_transition_tensor(self, X: np.ndarray) -> np.ndarray:
        T = len(X)
        out = np.zeros((T, self.n_states, self.n_states), dtype=float)
        for i in range(self.n_states):
            out[:, i, :] = self._row_log_probs(X, i)
        return out

    def _objective_and_grad(
        self,
        w_flat: np.ndarray,
        X1: np.ndarray,
        soft_targets: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        K = self.n_states
        W = w_flat.reshape(K, X1.shape[1])

        logits = X1 @ W.T
        logits = logits - logits.max(axis=1, keepdims=True)
        log_probs = logits - logsumexp(logits, axis=1, keepdims=True)
        probs = np.exp(log_probs)

        loss = -np.sum(soft_targets * log_probs)
        loss += 0.5 * self.l2 * np.sum(W[:, 1:] ** 2)

        grad = (probs - soft_targets).T @ X1
        grad[:, 1:] += self.l2 * W[:, 1:]

        return float(loss), grad.ravel()

    def fit(self, X: np.ndarray, xi: np.ndarray) -> None:
        X_next = X[1:]
        X1 = _add_intercept(X_next)

        for i in range(self.n_states):
            targets = xi[:, i, :]
            x0 = self.W[i].ravel().copy()

            def make_fun(targets_i: np.ndarray):
                def fun(w_flat: np.ndarray) -> Tuple[float, np.ndarray]:
                    return self._objective_and_grad(w_flat, X1, targets_i)
                return fun

            res = minimize(
                fun=make_fun(targets),
                x0=x0,
                jac=True,
                method="L-BFGS-B",
            )

            if res.success:
                self.W[i] = res.x.reshape(self.n_states, X1.shape[1])
            else:
                warnings.warn(
                    f"Transition optimization failed for prev_state={i}: {res.message}. "
                    "Keeping previous weights."
                )

        self._project_reference()


@dataclass
class IOHMMFitResult:
    log_likelihoods: List[float]
    converged: bool
    n_iter: int


class GaussianIOHMM:
    def __init__(
        self,
        n_states: int = 3,
        emission_ridge: float = 1e-4,
        transition_l2: float = 1e-3,
        max_iter: int = 100,
        tol: float = 1e-4,
        random_state: Optional[int] = 42,
        n_init: int = 10,
    ):
        if n_states < 2:
            raise ValueError("n_states must be >= 2")

        self.n_states = int(n_states)
        self.K = self.n_states
        self.emission_ridge = float(emission_ridge)
        self.transition_l2 = float(transition_l2)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.random_state = random_state
        self.n_init = int(n_init)

        self.feature_mean_: Optional[np.ndarray] = None
        self.feature_std_: Optional[np.ndarray] = None
        self.init_log_probs_: Optional[np.ndarray] = None

        self.emission_model: Optional[GaussianEmissionModel] = None
        self.transition_model: Optional[SoftmaxTransitionModel] = None
        self.is_fitted_: bool = False

        self.sigma2_floor_: float = 1e-4
        self.smoothed_posteriors_: Optional[np.ndarray] = None
        self.state_order_: Optional[np.ndarray] = None
        self.best_loglik_: Optional[float] = None
        self.bic_: Optional[float] = None
        self.icl_: Optional[float] = None

    @property
    def beta_(self) -> np.ndarray:
        if self.emission_model is None:
            raise RuntimeError("Model is not initialized.")
        return np.array([st.beta for st in self.emission_model.states])

    @property
    def sigma2_(self) -> np.ndarray:
        if self.emission_model is None:
            raise RuntimeError("Model is not initialized.")
        return np.array([st.sigma2 for st in self.emission_model.states])

    @property
    def pi_(self) -> np.ndarray:
        if self.init_log_probs_ is None:
            raise RuntimeError("Model is not initialized.")
        return np.exp(self.init_log_probs_)

    def _initialize(self, X: np.ndarray, y: np.ndarray, rng: Optional[np.random.Generator] = None) -> None:
        Xs, mean, std = _standardize_fit(X)
        self.feature_mean_ = mean
        self.feature_std_ = std

        self.init_log_probs_ = np.log(np.ones(self.n_states) / self.n_states)

        self.emission_model = GaussianEmissionModel(
            n_states=self.n_states,
            n_features=X.shape[1],
            ridge=self.emission_ridge,
            sigma2_floor=self.sigma2_floor_,
        )
        self.emission_model.initialize_from_quantiles(Xs, y)

        self.transition_model = SoftmaxTransitionModel(
            n_states=self.n_states,
            n_features=X.shape[1],
            l2=self.transition_l2,
        )

        if rng is not None:
            for st in self.emission_model.states:
                scale_beta = 0.1 * (np.abs(st.beta) + 1e-2)
                st.beta = st.beta + rng.normal(scale=scale_beta, size=st.beta.shape)
                st.sigma2 = max(float(st.sigma2 * rng.uniform(0.5, 2.0)), self.sigma2_floor_)
            scale_W = 0.1 * (np.abs(self.transition_model.W) + 1e-2)
            self.transition_model.W = self.transition_model.W + rng.normal(scale=scale_W)
            self.transition_model._project_reference()

    def _transform_X(self, X: np.ndarray) -> np.ndarray:
        if self.feature_mean_ is None or self.feature_std_ is None:
            raise RuntimeError("Model not initialized.")
        return _standardize_apply(X, self.feature_mean_, self.feature_std_)

    def _forward_backward(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.emission_model is None or self.transition_model is None or self.init_log_probs_ is None:
            raise RuntimeError("Model not initialized.")

        T = len(y)
        K = self.n_states

        Xs = self._transform_X(X)
        log_emiss = self.emission_model.log_likelihood_matrix(Xs, y)
        log_trans = self.transition_model.log_transition_tensor(Xs)

        log_alpha = np.full((T, K), -np.inf, dtype=float)
        log_beta = np.full((T, K), -np.inf, dtype=float)

        log_alpha[0] = self.init_log_probs_ + log_emiss[0]

        for t in range(1, T):
            for j in range(K):
                log_alpha[t, j] = log_emiss[t, j] + logsumexp(log_alpha[t - 1] + log_trans[t, :, j])

        loglik = float(logsumexp(log_alpha[-1]))
        log_beta[-1] = 0.0

        for t in range(T - 2, -1, -1):
            for i in range(K):
                log_beta[t, i] = logsumexp(log_trans[t + 1, i, :] + log_emiss[t + 1, :] + log_beta[t + 1, :])

        log_gamma = log_alpha + log_beta
        log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)
        gamma = np.exp(log_gamma)

        xi = np.zeros((T - 1, K, K), dtype=float)
        for t in range(1, T):
            log_xi_t = (
                log_alpha[t - 1][:, None]
                + log_trans[t]
                + log_emiss[t][None, :]
                + log_beta[t][None, :]
            )
            log_xi_t -= logsumexp(log_xi_t)
            xi[t - 1] = np.exp(log_xi_t)

        return loglik, gamma, xi, log_alpha, log_beta

    def _fit_single(
        self, X: np.ndarray, y: np.ndarray, rng: Optional[np.random.Generator] = None
    ) -> Optional[IOHMMFitResult]:
        self._initialize(X, y, rng=rng)

        log_likelihoods: List[float] = []
        converged = False

        for it in range(self.max_iter):
            loglik, gamma, xi, _, _ = self._forward_backward(X, y)
            log_likelihoods.append(loglik)

            Xs = self._transform_X(X)
            if self.emission_model is None:
                raise RuntimeError("Emission model missing during EM step.")
            if self.transition_model is None:
                raise RuntimeError("Transition model missing during EM step.")

            self.emission_model.fit(Xs, y, gamma)
            self.transition_model.fit(Xs, xi)

            self.init_log_probs_ = np.log(np.maximum(gamma[0], EPS))
            self.init_log_probs_ -= logsumexp(self.init_log_probs_)

            if it > 0:
                improvement = log_likelihoods[-1] - log_likelihoods[-2]
                if improvement < -self.tol:
                    warnings.warn(
                        f"EM iteration {it}: log-likelihood decreased by {-improvement:.6f}. "
                        "Discarding run."
                    )
                    return None
                if improvement < self.tol:
                    converged = True
                    break

        return IOHMMFitResult(
            log_likelihoods=log_likelihoods,
            converged=converged,
            n_iter=len(log_likelihoods),
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> IOHMMFitResult:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be 2D.")
        if y.ndim != 1:
            raise ValueError("y must be 1D.")
        if len(X) != len(y):
            raise ValueError("X and y must have same number of rows.")
        if len(y) < 20:
            raise ValueError("Need more observations to fit IOHMM.")
        if np.any(~np.isfinite(X)) or np.any(~np.isfinite(y)):
            raise ValueError("X and y must not contain NaN or Inf values.")

        self.is_fitted_ = False
        self.sigma2_floor_ = max(1e-4, 0.01 * float(np.var(y)))

        seed_seq = np.random.SeedSequence(self.random_state)
        best_result: Optional[IOHMMFitResult] = None
        best_loglik = -np.inf
        best_state: Optional[dict] = None

        for run in range(self.n_init):
            rng = np.random.default_rng(seed_seq.spawn(1)[0]) if run > 0 else None
            result = self._fit_single(X, y, rng=rng)
            if result is None:
                continue
            final_ll = result.log_likelihoods[-1]
            if final_ll > best_loglik:
                best_loglik = final_ll
                best_result = result
                best_state = {
                    "emission_model": self.emission_model,
                    "transition_model": self.transition_model,
                    "init_log_probs_": self.init_log_probs_,
                    "feature_mean_": self.feature_mean_,
                    "feature_std_": self.feature_std_,
                }

        if best_state is None or best_result is None:
            raise RuntimeError(f"All {self.n_init} EM runs failed (log-likelihood diverged).")

        self.emission_model = best_state["emission_model"]
        self.transition_model = best_state["transition_model"]
        self.init_log_probs_ = best_state["init_log_probs_"]
        self.feature_mean_ = best_state["feature_mean_"]
        self.feature_std_ = best_state["feature_std_"]
        self.best_loglik_ = best_loglik

        _, gamma, _, _, _ = self._forward_backward(X, y)
        self.smoothed_posteriors_ = gamma

        sigmas = np.array([st.sigma2 for st in self.emission_model.states])
        order = np.argsort(sigmas)
        self.state_order_ = order

        self.emission_model.states = [self.emission_model.states[k] for k in order]

        W = self.transition_model.W
        W_perm = W[order][:, order, :]
        W_perm = W_perm - W_perm[:, self.n_states - 1: self.n_states, :]
        self.transition_model.W = W_perm

        self.init_log_probs_ = self.init_log_probs_[order]
        self.init_log_probs_ -= logsumexp(self.init_log_probs_)
        self.smoothed_posteriors_ = self.smoothed_posteriors_[:, order]

        T = len(y)
        K = self.n_states
        n_features = X.shape[1]
        n_params = K * K * (n_features + 1) + K * (n_features + 1) + K
        self.bic_ = -2 * self.best_loglik_ + n_params * np.log(T)
        entropy = -float(np.sum(self.smoothed_posteriors_ * np.log(self.smoothed_posteriors_ + 1e-12)))
        self.icl_ = self.bic_ + 2 * entropy

        self.is_fitted_ = True
        return best_result

    def forward_filter(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        self._ensure_fitted()
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if self.emission_model is None or self.transition_model is None or self.init_log_probs_ is None:
            raise RuntimeError("Model not initialized.")

        Xs = self._transform_X(X)
        T = len(y)
        K = self.n_states

        log_emiss = self.emission_model.log_likelihood_matrix(Xs, y)
        log_trans = self.transition_model.log_transition_tensor(Xs)

        log_alpha = np.full((T, K), -np.inf, dtype=float)
        log_alpha[0] = self.init_log_probs_ + log_emiss[0]

        for t in range(1, T):
            for j in range(K):
                log_alpha[t, j] = log_emiss[t, j] + logsumexp(log_alpha[t - 1] + log_trans[t, :, j])

        return log_alpha

    def forecast(
        self,
        X_history: np.ndarray,
        y_history: np.ndarray,
        x_next: np.ndarray,
    ) -> Tuple[float, float, np.ndarray]:
        self._ensure_fitted()
        if self.emission_model is None or self.transition_model is None:
            raise RuntimeError("Model not initialized.")

        X_history = np.asarray(X_history, dtype=float)
        y_history = np.asarray(y_history, dtype=float)
        x_next = np.asarray(x_next, dtype=float).ravel()

        log_alpha = self.forward_filter(X_history, y_history)
        alpha_T = np.exp(log_alpha[-1] - logsumexp(log_alpha[-1]))

        x_next_s = _standardize_apply(x_next[None, :], self.feature_mean_, self.feature_std_)
        x_next_s1 = np.concatenate([[1.0], x_next_s[0]])

        log_trans_next = self.transition_model.log_transition_tensor(x_next_s)[0]
        p_next = alpha_T @ np.exp(log_trans_next)
        p_next = p_next / p_next.sum()

        mu_next = np.array([x_next_s1 @ st.beta for st in self.emission_model.states])
        sigma2 = np.array([st.sigma2 for st in self.emission_model.states])

        y_hat = float(p_next @ mu_next)
        var_hat = float(p_next @ (sigma2 + (mu_next - y_hat) ** 2))
        return y_hat, var_hat, p_next

    def predict_state_proba(self, X: np.ndarray, y: np.ndarray, smoothed: bool = True) -> np.ndarray:
        self._ensure_fitted()
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if smoothed:
            _, gamma, _, _, _ = self._forward_backward(X, y)
            return gamma
        log_alpha = self.forward_filter(X, y)
        log_alpha_norm = log_alpha - logsumexp(log_alpha, axis=1, keepdims=True)
        return np.exp(log_alpha_norm)

    def predict_states(self, X: np.ndarray, y: np.ndarray, smoothed: bool = True) -> np.ndarray:
        gamma = self.predict_state_proba(X, y, smoothed=smoothed)
        return np.argmax(gamma, axis=1)

    def viterbi(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        self._ensure_fitted()

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        Xs = self._transform_X(X)

        if self.emission_model is None:
            raise RuntimeError("Emission model not initialized.")
        if self.transition_model is None:
            raise RuntimeError("Transition model not initialized.")
        if self.init_log_probs_ is None:
            raise RuntimeError("Initial probabilities not initialized.")

        T = len(y)
        K = self.n_states

        log_emiss = self.emission_model.log_likelihood_matrix(Xs, y)
        log_trans = self.transition_model.log_transition_tensor(Xs)

        delta = np.full((T, K), -np.inf, dtype=float)
        psi = np.zeros((T, K), dtype=int)

        delta[0] = self.init_log_probs_ + log_emiss[0]

        for t in range(1, T):
            for j in range(K):
                vals = delta[t - 1] + log_trans[t, :, j]
                psi[t, j] = int(np.argmax(vals))
                delta[t, j] = np.max(vals) + log_emiss[t, j]

        states = np.zeros(T, dtype=int)
        states[-1] = int(np.argmax(delta[-1]))
        for t in range(T - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]
        return states

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        self._ensure_fitted()
        loglik, _, _, _, _ = self._forward_backward(np.asarray(X, float), np.asarray(y, float))
        return loglik

    def make_results_frame(
        self,
        dates: pd.DatetimeIndex,
        X: np.ndarray,
        y: np.ndarray,
        state_labels: Optional[Sequence[str]] = None,
        use_viterbi: bool = True,
        smoothed: bool = True,
    ) -> pd.DataFrame:
        gamma = self.predict_state_proba(X, y, smoothed=smoothed)
        if use_viterbi and smoothed:
            states = self.viterbi(X, y)
        else:
            states = np.argmax(gamma, axis=1)

        if state_labels is None:
            state_labels = [f"state_{k}" for k in range(self.n_states)]

        df = pd.DataFrame(index=dates)
        df["y"] = y
        df["state"] = states
        for k, lab in enumerate(state_labels):
            df[f"p_{lab}"] = gamma[:, k]
        return df

    def _ensure_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError("Model is not fitted.")
