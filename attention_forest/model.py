import logging
import numpy as np
import cvxpy
from abc import ABC, abstractmethod
from typing import NamedTuple, Optional, Tuple
from sklearn.model_selection import train_test_split
from .forests import *
from scipy.special import softmax as _softmax
import cvxpy as cp
import scipy
from time import time
from sklearn.preprocessing import OneHotEncoder


class ClfRegHot:
    @abstractmethod
    def fit(self, X, y) -> 'ClfRegHot':
        pass

    @abstractmethod
    def refit(self, X, y) -> 'ClfRegHot':
        """Fit to the new dataset with warm start."""
        pass

    @abstractmethod
    def optimize_weights(self, X, y) -> 'ClfRegHot':
        """Optimize weights for the new dataset."""
        pass

    @abstractmethod
    def predict(self, X) -> np.ndarray:
        pass

    @abstractmethod
    def predict_original(self, X) -> np.ndarray:
        pass


class AFParams(NamedTuple):
    """Parameters of Attention Forest."""
    kind: ForestKind
    task: TaskType
    loss_ord: int = 2  # loss order
    eps: Optional[int] = None
    discount: Optional[float] = None
    forest: dict = {}


class EAFParams(AFParams):
    """Parameters of Epsilon-Attention Forest."""
    # eps: float = 0.1
    # tau: float = 1.0
    def __new__(cls, *args, eps: float = 0.1, tau: float = 1.0, **kwargs):
        self = super(EAFParams, cls).__new__(cls, *args, eps=eps, **kwargs)
        self.tau = tau
        return self


class FWAFParams(AFParams):
    """Parameters of Feature-Weighted Attention Forest."""
    def __new__(cls, *args, no_temp: bool = False, **kwargs):
        self = super(FWAFParams, cls).__new__(cls, *args, **kwargs)
        self.no_temp = no_temp
        return self


class LeafData(NamedTuple):
    xs: np.ndarray
    y: np.ndarray


def _prepare_leaf_data_fast(xs, y, leaf_ids, estimators, multiplier=1.0):
    max_leaf_id = max(map(lambda e: e.tree_.node_count, estimators))
    y_len = 1 if y.ndim == 1 else y.shape[1]
    result_x = np.full((len(estimators), max_leaf_id + 1, xs.shape[1]), np.nan, dtype=np.float32)
    result_y = np.full((len(estimators), max_leaf_id + 1, y_len), np.nan, dtype=np.float32)
    for tree_id in range(len(estimators)):
        for leaf_id in range(estimators[tree_id].tree_.node_count + 1):
            mask = (leaf_ids[:, tree_id] == leaf_id)
            masked_xs = xs[mask]
            masked_y = y[mask]
            if mask.any():
                result_x[tree_id, leaf_id] = masked_xs.mean(axis=0)
                result_y[tree_id, leaf_id] = masked_y.mean(axis=0) * multiplier
    return result_x, result_y


def _convert_labels_to_probas(y, encoder=None):
    if y.ndim == 2 and y.shape[1] >= 2:
        return y, encoder
    if encoder is None:
        encoder = OneHotEncoder()
        y = encoder.fit_transform(y.reshape((-1, 1)))
    else:
        y = encoder.transform(y.reshape((-1, 1)))
    return y, encoder


class AttentionForest(ClfRegHot):
    def __init__(self, params: AFParams):
        self.params = params
        self.forest = None
        self._after_init()

    def _after_init(self):
        self.onehot_encoder = None

    def _make_leaf_data(self, leaf_id, tree_leaf_ids, xs, y):
        mask = (tree_leaf_ids == leaf_id)
        masked_xs = xs[mask]
        masked_y = y[mask]
        return LeafData(xs=masked_xs, y=masked_y)

    def _preprocess_target(self, y):
        if self.params.task == TaskType.CLASSIFICATION:
            y, self.onehot_encoder = _convert_labels_to_probas(y, self.onehot_encoder)
        return y


    def fit(self, X, y) -> 'AttentionForest':
        forest_cls = FORESTS[ForestType(self.params.kind, self.params.task)]
        self.forest = forest_cls(**self.params.forest)
        # print("Start fitting Random forest")
        start_time = time()
        self.forest.fit(X, y)
        end_time = time()
        # print("Random forest fit time:", end_time - start_time)
        # store training X and y
        self.training_xs = X.copy()
        self.training_y = self._preprocess_target(y.copy())
        # store leaf id for each point in X
        start_time = time()
        self.training_leaf_ids = self.forest.apply(self.training_xs)
        end_time = time()
        # print("Random forest apply time:", end_time - start_time)
        # make a tree-leaf-points correspondence
        # print("Generating leaves data")
        start_time = time()
        # multiplier = self.forest.learning_rate if self.params.kind.need_add_init() else 1.0
        multiplier = 1.0
        if hasattr(self.forest, 'get_leaf_data'):
            self.leaf_data_x, self.leaf_data_y = self.forest.get_leaf_data()
        else:
            self.leaf_data_x, self.leaf_data_y = _prepare_leaf_data_fast(
                self.training_xs,
                self.training_y,
                self.training_leaf_ids,
                self.forest.estimators_,
                multiplier=multiplier
            )
        end_time = time()
        # print("Leaf generation time:", end_time - start_time)
        self.tree_weights = np.ones(self.forest.n_estimators)
        self.static_weights = np.ones(self.forest.n_estimators) / self.forest.n_estimators
        return self

    def optimize_weights(self, X, y_orig) -> 'AttentionForest':
        raise NotImplementedError("Use EpsAttentionForest or FeatureWeightedAttentionForest instead.")

    def refit(self, X, y) -> 'AttentionForest':
        assert self.forest is not None, "Need to fit before refit"
        # update leaves?
        raise NotImplementedError()
        # find tree weights
        self.optimize_weights(X, y)
        return self

    def _get_dynamic_weights_y(self, X) -> Tuple[np.ndarray, np.ndarray]:
        leaf_ids = self.forest.apply(X)
        all_dynamic_weights = []
        all_y = []
        for cur_x, cur_leaf_ids in zip(X, leaf_ids):
            tree_dynamic_weights = []
            tree_dynamic_y = []
            for cur_tree_id, cur_leaf_id in enumerate(cur_leaf_ids):
                leaf_mean_x = self.leaf_data_x[cur_tree_id][cur_leaf_id]
                leaf_mean_y = self.leaf_data_y[cur_tree_id][cur_leaf_id]
                tree_dynamic_weight = -0.5 * np.linalg.norm(cur_x - leaf_mean_x, 2) ** 2.0
                if self.params.discount is not None:
                    tree_dynamic_weight *= self.params.discount ** cur_tree_id
                tree_dynamic_weights.append(tree_dynamic_weight)
                tree_dynamic_y.append(leaf_mean_y)
            tree_dynamic_weights = np.array(tree_dynamic_weights)
            tree_dynamic_y = np.array(tree_dynamic_y)
            all_dynamic_weights.append(tree_dynamic_weights)
            all_y.append(tree_dynamic_y)
        all_dynamic_weights = np.array(all_dynamic_weights)
        all_y = np.array(all_y)
        return all_dynamic_weights, all_y

    def predict(self, X) -> np.ndarray:
        assert self.forest is not None, "Need to fit before predict"
        all_dynamic_weights, all_y = self._get_dynamic_weights_y(X)
        weights = _softmax(all_dynamic_weights * self.tree_weights[np.newaxis], axis=1)
        if self.params.eps is not None:
            mixed_weights = (1.0 - self.params.eps) * weights + self.params.eps * self.static_weights
        else:
            mixed_weights = weights
        mixed_weights = mixed_weights[..., np.newaxis]
        predictions = np.sum(mixed_weights * all_y, axis=1)
        if self.params.kind.need_add_init():
            predictions += self.forest.init_.predict(X)
        return predictions

    def predict_original(self, X):
        if self.params.task == TaskType.REGRESSION:
            return self.forest.predict(X)
        elif self.params.task == TaskType.CLASSIFICATION:
            return self.forest.predict_proba(X)
        raise ValueError(f'Unsupported task type in predict_original: "{self.params.task}"')


class EpsAttentionForest(AttentionForest):
    def __init__(self, params: EAFParams):
        self.params = params
        self.forest = None
        self.w = None
        self._after_init()

    def fit(self, X, y):
        super().fit(X, y)
        self.w = np.ones(self.forest.n_estimators) / self.forest.n_estimators

    def optimize_weights(self, X, y_orig) -> 'EpsAttentionForest':
        assert self.forest is not None, "Need to fit before weights optimization"
        dynamic_weights, dynamic_y = self._get_dynamic_weights_y(X)
        static_weights = cp.Variable((1, self.forest.n_estimators))

        bias = 0.0
        if self.params.kind.need_add_init():
            bias = self.forest.init_.predict(X)
        y = y_orig.copy()
        y -= bias

        if dynamic_y.shape[2] == 1:
            dynamic_y = dynamic_y[..., 0]
            mixed_weights = (1.0 - self.params.eps) * dynamic_weights + self.params.eps * static_weights
        else:
            # y0: y0t0_0 y0t0_1 ... y0t0_d | y0t1_0 y0t1_1 ... y0t1_d | ... | y0tT_0 ... y0tT_d
            # loss = sum_i sum_j (sum_k yitk_j - yi_j)
            # dynamic_y = dynamic_y.reshape((dynamic_y.shape[0], -1))
            # swap 1 and 2 axes and merge "sample" and "feature" axes
            n_trees = dynamic_y.shape[1]
            n_outs = dynamic_y.shape[2]
            dynamic_y = np.transpose(dynamic_y, (0, 2, 1)).reshape((-1, dynamic_y.shape[1]))
            y = y.reshape((-1))
            # dynamic_weights shape: (n_samples, n_trees)
            # repeat dynamic weights for each output
            dynamic_weights = np.tile(dynamic_weights[:, np.newaxis, :], (1, n_outs, 1)).reshape((-1, n_trees))
            mixed_weights = (1.0 - self.params.eps) * dynamic_weights + self.params.eps * static_weights
            y, self.onehot_encoder = _convert_labels_to_probas(y, self.onehot_encoder)
            y = y.toarray().ravel()
            print("Shapes:", mixed_weights.shape, dynamic_y.shape)
        loss_terms = cp.sum(cp.multiply(mixed_weights, dynamic_y), axis=1) - y
        if self.params.loss_ord == 1:
            min_obj = cp.sum(cp.abs(loss_terms))
        elif self.params.loss_ord == 2:
            min_obj = cp.sum_squares(loss_terms)
        else:
            raise ValueError(f'Wrong loss order: {self.params.loss_ord}')
        problem = cp.Problem(cp.Minimize(min_obj),
            [
                static_weights >= 0,
                cp.sum(static_weights, axis=1) == 1
            ]
        )

        try:
            loss_value = problem.solve()
        except Exception as ex:
            logging.warning(f"Solver error: {ex}")

        if static_weights.value is None:
            logging.info(f"Can't solve problem with OSQP. Trying another solver...")
            loss_value = problem.solve(solver=cvxpy.SCS)

        if static_weights.value is None:
            logging.warn(f"Weights optimization error (eps={self.params.eps}). Using default values.")
        else:
            self.w = static_weights.value.copy().reshape((-1,))
        return self

    def _get_dynamic_weights_y(self, X) -> Tuple[np.ndarray, np.ndarray]:
        leaf_ids = self.forest.apply(X)
        all_dynamic_weights = []
        all_y = []
        for cur_x, cur_leaf_ids in zip(X, leaf_ids):
            tree_dynamic_weights = []
            tree_dynamic_y = []
            for cur_tree_id, cur_leaf_id in enumerate(cur_leaf_ids):
                leaf_mean_x = self.leaf_data_x[cur_tree_id][cur_leaf_id]
                leaf_mean_y = self.leaf_data_y[cur_tree_id][cur_leaf_id]
                tree_dynamic_weight = -0.5 * np.linalg.norm(cur_x - leaf_mean_x, 2) ** 2.0
                if self.params.discount is not None:
                    tree_dynamic_weight *= self.params.discount ** cur_tree_id
                tree_dynamic_weights.append(tree_dynamic_weight)
                tree_dynamic_y.append(leaf_mean_y)
            tree_dynamic_weights = _softmax(np.array(tree_dynamic_weights) * self.params.tau)
            tree_dynamic_y = np.array(tree_dynamic_y)
            all_dynamic_weights.append(tree_dynamic_weights)
            all_y.append(tree_dynamic_y)
        all_dynamic_weights = np.array(all_dynamic_weights)
        all_y = np.array(all_y)
        return all_dynamic_weights, all_y

    def predict(self, X) -> np.ndarray:
        assert self.forest is not None, "Need to fit before predict"
        all_dynamic_weights, all_y = self._get_dynamic_weights_y(X)
        mixed_weights = (1.0 - self.params.eps) * all_dynamic_weights + self.params.eps * self.w
        mixed_weights = mixed_weights[..., np.newaxis]
        predictions = np.sum(mixed_weights * all_y, axis=1)
        if self.params.kind.need_add_init():
            predictions += self.forest.init_.predict(X)[:, np.newaxis]
        return predictions


class FeatureWeightedAttentionForest(AttentionForest):
    def __init__(self, params: FWAFParams):
        self.params = params
        self.forest = None
        self._after_init()

    def fit(self, X, y) -> 'AttentionForest':
        super().fit(X, y)
        n_features = X.shape[1]
        self.feature_weights = np.ones(n_features)
        return self

    def optimize_weights(self, X, y) -> 'AttentionForest':
        assert self.forest is not None, "Need to fit before weights optimization"
        y = self._preprocess_target(y)
        dynamic_weights, dynamic_y = self._get_dynamic_weights_y(X)

        if dynamic_y.shape[2] == 1:
            dynamic_y = dynamic_y[..., 0]

        w_init = np.ones(self.forest.n_estimators)
        n_features = X.shape[1]
        feature_weights_init = np.ones(n_features)
        if self.params.eps is not None:
            static_weights_init = np.ones(self.forest.n_estimators) / self.forest.n_estimators

        def _model(cur_feature_weights, cur_w, cur_static_weights):
            new_dyn_weights = -0.5 * np.linalg.norm(dynamic_weights / cur_feature_weights, 2, axis=-1) ** 2.0
            alphas = new_dyn_weights * np.abs(cur_w)
            alphas_softmax = _softmax(alphas, axis=1)
            if self.params.eps is not None:
                static_softmax = _softmax(cur_static_weights, axis=0)
                mixed_weights = (1.0 - self.params.eps) * alphas_softmax + self.params.eps * static_softmax
            else:
                mixed_weights = alphas_softmax
            if dynamic_y.ndim == 3:
                mixed_weights = mixed_weights[..., np.newaxis]
            return np.sum(
                np.multiply(mixed_weights, dynamic_y),
                axis=1
            )

        def _loss(cur_preds, y_true):
            loss_terms = cur_preds - y_true
            if self.params.loss_ord == 1:
                loss = np.sum(np.abs(loss_terms))
            elif self.params.loss_ord == 2:
                if loss_terms.ndim == 2:
                    loss = np.sum(np.linalg.norm(loss_terms, 2, axis=1) ** 2)
                else:
                    loss = np.sum(loss_terms ** 2)
            else:
                raise ValueError(f'Wrong loss order: {self.params.loss_ord}')
            return loss

        # print("Dynamic_y", dynamic_y.shape)
        opt_params = [w_init, feature_weights_init]

        if self.params.eps is not None:
            opt_params.append(static_weights_init)
        if self.params.no_temp:
            # don't optimize `w`
            params_to_optimize = opt_params[1:]
        else:
            params_to_optimize = opt_params
        params_to_optimize_lens = list(map(len, params_to_optimize))

        def _min_fn(cur_merged_weights):
            # split weights
            cur_weights = [
                cur_merged_weights[prev_cum_len:prev_cum_len + cur_len]
                for prev_cum_len, cur_len in zip([0] + list(np.cumsum(params_to_optimize_lens))[:-1], params_to_optimize_lens)
            ]
            if self.params.no_temp:
                cur_tree_weights = w_init
                idx = 0
            else:
                cur_tree_weights = cur_weights[0]
                idx = 1
            cur_feature_weights = cur_weights[idx]
            if self.params.eps is not None:
                cur_static_weights = cur_weights[idx + 1]
            cur_preds = _model(cur_feature_weights, cur_tree_weights, cur_static_weights)
            loss = _loss(cur_preds, y)
            return loss

        params_to_optimize = np.concatenate(params_to_optimize, axis=0)
        print(params_to_optimize.shape)
        result = scipy.optimize.minimize(_min_fn, params_to_optimize, method='L-BFGS-B', jac=False)
        optimal_weights = [
            result.x[prev_cum_len:prev_cum_len + cur_len]
            for prev_cum_len, cur_len in zip([0] + list(np.cumsum(params_to_optimize_lens))[:-1], params_to_optimize_lens)
        ]
        
        if self.params.no_temp:
            idx = 0
        else:
            self.tree_weights = optimal_weights[0].copy()
            idx = 1
        self.feature_weights = optimal_weights[idx].copy()
        if self.params.eps is not None:
            self.static_weights = _softmax(optimal_weights[idx + 1].copy())

        return self

    
    def optimize_weights_torch(self, X, y) -> 'AttentionForest':
        raise Exception("")
        assert self.forest is not None, "Need to fit before weights optimization"
        dynamic_weights, dynamic_y = self._get_dynamic_weights_y(X)

        if dynamic_y.shape[2] == 1:
            dynamic_y = dynamic_y[..., 0]

        w_init = np.ones(self.forest.n_estimators)
        torch_w = torch.tensor(
            w_init,
            dtype=torch.float32,
            requires_grad=True
        )
        
        n_features = X.shape[1]
        feature_weights_init = np.ones(n_features)
        torch_feature_weights = torch.tensor(
            feature_weights_init,
            dtype=torch.float32,
            requires_grad=(not self.params.no_temp)
        )

        if self.params.eps is not None:
            torch_static_weights = torch.full(
                (self.forest.n_estimators,),
                1 / self.forest.n_estimators,
                dtype=torch.float32,
                requires_grad=True
            )
        
        def _model(torch_dyn_weights, torch_dyn_y):
            new_dyn_weights = -0.5 * torch.norm(torch_dyn_weights / torch_feature_weights, 2, dim=-1) ** 2.0
            alphas = new_dyn_weights * torch.abs(torch_w)
            alphas_softmax = torch.nn.functional.softmax(alphas, dim=1)
            if self.params.eps is not None:
                static_softmax = torch.nn.functional.softmax(torch_static_weights, dim=0)
                mixed_weights = (1.0 - self.params.eps) * alphas_softmax + self.params.eps * static_softmax
            else:
                mixed_weights = alphas_softmax
            return torch.sum(
                torch.multiply(mixed_weights, torch_dyn_y),
                axis=1
            )

        def _loss(torch_preds, torch_y):
            loss_terms = torch_preds - torch_y
            if self.params.loss_ord == 1:
                loss = torch.sum(torch.abs(loss_terms))
            elif self.params.loss_ord == 2:
                loss = torch.sum(loss_terms ** 2)
            else:
                raise ValueError(f'Wrong loss order: {self.params.loss_ord}')
            return loss

        print("Dynamic_y", dynamic_y.shape)
        opt_params = [torch_w, torch_feature_weights]
        if self.params.eps is not None:
            opt_params.append(torch_static_weights)
        if self.params.no_temp:
            # don't optimize `torch_w`
            params_to_optimize = opt_params[1:]
        else:
            params_to_optimize = opt_params
        adam_optim = torch.optim.Adam(params_to_optimize, lr=1e-4, weight_decay=1e-5)
        optimal_weights = optimize(
            _model,
            _loss,
            adam_optim,
            opt_params,
            [  # inputs
                dynamic_weights,
                dynamic_y
            ],
            y,
            n_epochs=100,
            n_batches=2,
        )
        self.tree_weights = optimal_weights[0].copy()
        self.feature_weights = optimal_weights[1].copy()
        if self.params.eps is not None:
            self.static_weights = _softmax(optimal_weights[2].copy())

        return self

    def _get_dynamic_weights_y(self, X) -> Tuple[np.ndarray, np.ndarray]:
        leaf_ids = self.forest.apply(X)
        all_dynamic_weights = []
        all_y = []
        for cur_x, cur_leaf_ids in zip(X, leaf_ids):
            tree_dynamic_weights = []
            tree_dynamic_y = []
            for cur_tree_id, cur_leaf_id in enumerate(cur_leaf_ids):
                leaf_mean_x = self.leaf_data_x[cur_tree_id][cur_leaf_id]
                leaf_mean_y = self.leaf_data_y[cur_tree_id][cur_leaf_id]
                # tree_dynamic_weight = -0.5 * np.linalg.norm(cur_x - leaf_mean_x, 2) ** 2.0
                # NOTE: here `tree_dynamic_weight` are not actually weights
                tree_dynamic_weight = cur_x - leaf_mean_x
                tree_dynamic_weights.append(tree_dynamic_weight)
                tree_dynamic_y.append(leaf_mean_y)
            tree_dynamic_weights = np.array(tree_dynamic_weights)
            tree_dynamic_y = np.array(tree_dynamic_y)
            all_dynamic_weights.append(tree_dynamic_weights)
            all_y.append(tree_dynamic_y)
        all_dynamic_weights = np.array(all_dynamic_weights)
        all_y = np.array(all_y)
        return all_dynamic_weights, all_y

    def predict(self, X) -> np.ndarray:
        assert self.forest is not None, "Need to fit before predict"
        all_dynamic_weights, all_y = self._get_dynamic_weights_y(X)
        print("  all_dynamic_weights shape:", all_dynamic_weights.shape, self.feature_weights.shape)
        all_dynamic_weights = -0.5 * np.linalg.norm(all_dynamic_weights / self.feature_weights, 2, axis=-1) ** 2.0
        weights = _softmax(all_dynamic_weights * self.tree_weights[np.newaxis], axis=1)
        if self.params.eps is not None:
            mixed_weights = (1.0 - self.params.eps) * weights + self.params.eps * self.static_weights
        else:
            mixed_weights = weights
        mixed_weights = mixed_weights[..., np.newaxis]
        predictions = np.sum(mixed_weights * all_y, axis=1)
        return predictions

    def predict_original(self, X):
        if self.params.task == TaskType.REGRESSION:
            return self.forest.predict(X)
        elif self.params.task == TaskType.CLASSIFICATION:
            return self.forest.predict_proba(X)
        raise ValueError(f'Unsupported task type in predict_original: "{self.params.task}"')



def fit_forest_split(X, y, params: AFParams, pre_size: float = 0.75, seed: Optional[int] = None):
    X_pre, X_post, y_pre, y_post = train_test_split(X, y, train_size=pre_size, random_state=seed)
    model = AttentionForest(params)
    model.fit(X_pre, y_pre)
    # model.refit(X_post, y_post)
    model.optimize_weights(X_post, y_post)
    return model

