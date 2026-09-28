"""
utils/shap_patch.py
===================
Runtime patch for SHAP + XGBoost 3.x compatibility.

In XGBoost 3.x, learner_model_param['base_score'] can be serialized as
an array string like '[5E-1]'. When SHAP's XGBTreeModelLoader parses it
via float(learner_model_param['base_score']), Python throws:
    ValueError: could not convert string to float: '[5E-1]'

This utility monkeypatches XGBTreeModelLoader to strip bracket characters
before float conversion if needed.
"""

import logging

logger = logging.getLogger(__name__)

_PATCH_APPLIED = False


def ensure_shap_xgboost_compatibility() -> None:
    """
    Ensure SHAP's TreeExplainer can load XGBoost 3.x models without
    failing on array-formatted base_score strings.
    """
    global _PATCH_APPLIED
    if _PATCH_APPLIED:
        return

    try:
        import shap.explainers._tree as tree_module

        orig_init = tree_module.XGBTreeModelLoader.__init__

        # Check if the class needs patching
        def patched_init(self, xgb_model):
            try:
                orig_init(self, xgb_model)
            except ValueError as exc:
                if "could not convert string to float" in str(exc):
                    # Re-run with sanitized base_score in UBJSON / model params
                    import io, numpy as np, scipy.special
                    from shap.explainers._tree import _check_xgboost_version, decode_ubjson_buffer
                    import xgboost as xgb

                    _check_xgboost_version(xgb.__version__)
                    model: xgb.Booster = xgb_model

                    raw = xgb_model.save_raw(raw_format="ubj")
                    with io.BytesIO(raw) as fd:
                        jmodel = decode_ubjson_buffer(fd)

                    learner = jmodel["learner"]
                    learner_model_param = learner["learner_model_param"]
                    objective = learner["objective"]
                    booster = learner["gradient_booster"]

                    n_classes = max(int(learner_model_param["num_class"]), 1)
                    n_targets = max(int(learner_model_param["num_target"]), 1)
                    n_targets = max(n_targets, n_classes)

                    if "gbtree" in booster and "model" not in booster:
                        booster = booster["gbtree"]

                    if booster["model"].get("iteration_indptr", None) is not None:
                        iteration_indptr = np.asarray(booster["model"]["iteration_indptr"], dtype=np.int32)
                        diff = np.diff(iteration_indptr)
                    else:
                        n_parallel_trees = int(booster["model"]["gbtree_model_param"]["num_parallel_tree"])
                        diff = np.repeat(n_targets * n_parallel_trees, model.num_boosted_rounds())

                    if np.any(diff != diff[0]):
                        raise ValueError("vector-leaf is not yet supported.:", diff)

                    self.n_trees_per_iter = int(diff[0])
                    self.n_targets = n_targets

                    raw_bs = str(learner_model_param["base_score"]).strip("[]")
                    base_score = float(raw_bs)
                    self.base_score = base_score
                    assert self.n_trees_per_iter > 0

                    self.name_obj = objective["name"]
                    self.name_gbm = booster["name"]

                    if self.name_obj in ("binary:logistic", "reg:logistic"):
                        self.base_score = scipy.special.logit(base_score)
                    elif self.name_obj in ("reg:gamma", "reg:tweedie", "count:poisson", "survival:cox", "survival:aft"):
                        self.base_score = np.log(self.base_score)

                    self.num_feature = int(learner_model_param["num_feature"])
                    self.num_class = int(learner_model_param["num_class"])

                    trees = booster["model"]["trees"]
                    self.num_trees = len(trees)

                    self.node_parents = []
                    self.node_cleft = []
                    self.node_cright = []
                    self.node_sindex = []
                    self.children_default = []
                    self.sum_hess = []
                    self.values = []
                    self.thresholds = []
                    self.features = []
                    self.split_types = []
                    self.categories = []

                    feature_types = model.feature_types
                    if feature_types is not None:
                        cat_feature_indices = np.where(np.asarray(feature_types) == "c")[0]
                        self.cat_feature_indices = cat_feature_indices if len(cat_feature_indices) > 0 else None
                    else:
                        self.cat_feature_indices = None

                    def to_integers(data):
                        return np.asanyarray(data, dtype=np.uint8)

                    for i in range(self.num_trees):
                        tree = trees[i]
                        parents = np.asarray(tree["parents"])
                        self.node_parents.append(parents)
                        self.node_cleft.append(np.asarray(tree["left_children"], dtype=np.int32))
                        self.node_cright.append(np.asarray(tree["right_children"], dtype=np.int32))
                        self.node_sindex.append(np.asarray(tree["split_indices"], dtype=np.uint32))

                        base_weight = np.asarray(tree["base_weights"], dtype=np.float32)
                        if base_weight.size != self.node_cleft[-1].size:
                            raise ValueError("vector-leaf is not yet supported.")

                        default_left = to_integers(tree["default_left"])
                        default_child = np.where(default_left == 1, self.node_cleft[-1], self.node_cright[-1]).astype(np.int64)
                        self.children_default.append(default_child)
                        self.sum_hess.append(np.asarray(tree["sum_hessian"], dtype=np.float64))

                        is_leaf = self.node_cleft[-1] == -1
                        split_cond = np.asarray(tree["split_conditions"], dtype=np.float32)
                        leaf_weight = np.where(is_leaf, split_cond, 0.0)
                        thresholds = np.where(is_leaf, 0.0, split_cond)
                        thresholds = np.where(is_leaf, 0.0, np.nextafter(thresholds, -np.float32(np.inf)))

                        self.values.append(leaf_weight.reshape(leaf_weight.size, 1))
                        self.thresholds.append(thresholds)
                        self.features.append(np.asarray(tree["split_indices"], dtype=np.int32))
                else:
                    raise

        tree_module.XGBTreeModelLoader.__init__ = patched_init
        _PATCH_APPLIED = True
        logger.debug("SHAP XGBTreeModelLoader compatibility patch registered.")
    except Exception as exc:
        logger.debug("Failed to apply SHAP compatibility patch: %s", exc)
