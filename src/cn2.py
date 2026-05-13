from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer

BIN_LABELS = ["low", "middle", "high"]


def _majority_class(y_sub: np.ndarray) -> Tuple[int, float]:
    if y_sub.size == 0:
        return 0, 0.0
    uniq, cnt = np.unique(y_sub, return_counts=True)
    k = int(cnt.argmax())
    return int(uniq[k]), float(cnt[k] / y_sub.size)


def _entropy(y_sub: np.ndarray) -> float:
    if y_sub.size <= 1:
        return 0.0
    _, cnt = np.unique(y_sub, return_counts=True)
    p = cnt / cnt.sum()
    return float(-np.sum(p * np.log2(np.clip(p, 1e-12, None))))


def covers(complex_conj: frozenset, xd_row: np.ndarray) -> bool:
    for a, v in complex_conj:
        if int(xd_row[a]) != int(v):
            return False
    return True


def _indices_matching(complex_conj: frozenset, Xd: np.ndarray, pool: np.ndarray) -> np.ndarray:
    matched = []
    for i in pool:
        if covers(complex_conj, Xd[int(i)]):
            matched.append(int(i))
    return np.array(matched, dtype=np.int64)


@dataclass
class CN2Rule:
    conditions: Tuple[Tuple[int, int], ...]
    predicted_class: int


class CN2Classifier:
    def __init__(
        self,
        n_bins: int = 3,
        beam_width: int = 6,
        max_conditions: int = 5,
        min_rule_coverage: int = 4,
        random_state: int = 42,
        test_fraction: float = 0.25,
    ):
        self.n_bins = n_bins
        self.beam_width = beam_width
        self.max_conditions = max_conditions
        self.min_rule_coverage = min_rule_coverage
        self.random_state = random_state
        self.test_fraction = test_fraction
        self.feature_names = [
            "Sepal length (cm)",
            "Sepal width (cm)",
            "Petal length (cm)",
            "Petal width (cm)",
        ]
        self.feature_names_short = ["Sepal L", "Sepal W", "Petal L", "Petal W"]
        self.class_names = ["Setosa", "Versicolor", "Virginica"]
        self.y_train_: np.ndarray = np.array([], dtype=np.int64)
        self.default_class_ = 0
        self.rule_stats_: list[dict] = []

    def fit_builtin_iris(self) -> dict:
        data = load_iris()
        X = data.data.astype(np.float64)
        y = data.target.astype(np.int64)
        X_train, _, y_train, _ = train_test_split(
            X,
            y,
            test_size=self.test_fraction,
            stratify=y,
            random_state=self.random_state,
        )
        self.fit_arrays(X_train, y_train)
        return {
            "n_samples_train": int(X_train.shape[0]),
            "n_classes": len(self.class_names),
            "n_features": int(X_train.shape[1]),
        }

    def fit_arrays(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X_train_ = X.copy()
        self.y_train_ = y.copy().astype(np.int64)
        self.default_class_ = int(np.argmax(np.bincount(self.y_train_, minlength=int(self.y_train_.max()) + 1)))

        self.discretizer = KBinsDiscretizer(
            n_bins=self.n_bins,
            encode="ordinal",
            strategy="quantile",
            quantile_method="averaged_inverted_cdf",
        )
        Xd = self.discretizer.fit_transform(X).astype(np.int64)
        self.edges_ = [np.array(e) for e in self.discretizer.bin_edges_]

        pool = np.arange(len(y), dtype=np.int64)
        self.rules_ = self._sequential_cover(Xd, y, pool)
        self.rule_stats_ = self._compute_rule_stats(Xd, y)

    def _compute_rule_stats(self, Xd: np.ndarray, y: np.ndarray) -> list[dict]:
        """Статистика first-match покриття/точності для кожного правила на train."""
        stats = [{"coverage": 0, "correct": 0, "precision": 0.0} for _ in self.rules_]
        for i in range(Xd.shape[0]):
            yi = int(y[i])
            for rid, rule in enumerate(self.rules_):
                if covers(frozenset(rule.conditions), Xd[i]):
                    stats[rid]["coverage"] += 1
                    if int(rule.predicted_class) == yi:
                        stats[rid]["correct"] += 1
                    break
        for row in stats:
            cov = int(row["coverage"])
            corr = int(row["correct"])
            row["precision"] = float(corr / cov) if cov > 0 else 0.0
        return stats

    def _score_complex(self, Xd: np.ndarray, y: np.ndarray, E: np.ndarray, C: frozenset) -> tuple[float, int] | None:
        sel = _indices_matching(C, Xd, E)
        if sel.size < self.min_rule_coverage:
            return None
        y_sub = y[sel]
        cls, purity = _majority_class(y_sub)
        entropy = _entropy(y_sub)
        weighted = purity - 0.08 * entropy + 0.002 * sel.size
        return float(weighted), int(cls)

    def _find_best_rule(
        self,
        Xd: np.ndarray,
        y: np.ndarray,
        E: np.ndarray,
    ) -> Tuple[frozenset, int] | None:
        beam: List[frozenset] = [frozenset()]
        global_best_C: frozenset | None = None
        global_best_score = -1e18
        global_best_cls = self.default_class_

        for _ in range(self.max_conditions):
            scored_beam_candidates: List[Tuple[float, frozenset, int]] = []

            for C in beam:
                sc = self._score_complex(Xd, y, E, C)
                if sc is None:
                    continue
                w, pcl = sc
                if w > global_best_score:
                    global_best_score = w
                    global_best_C = C
                    global_best_cls = pcl

                sel = _indices_matching(C, Xd, E)
                if sel.size < self.min_rule_coverage:
                    continue
                used_attrs = {a for a, _ in C}
                for a in range(Xd.shape[1]):
                    if a in used_attrs:
                        continue
                    for v in np.unique(Xd[sel, a]):
                        C2 = C | frozenset({(a, int(v))})
                        sc2 = self._score_complex(Xd, y, E, C2)
                        if sc2 is None:
                            continue
                        w2, pc2 = sc2
                        scored_beam_candidates.append((w2, C2, pc2))

            if not scored_beam_candidates:
                break

            scored_beam_candidates.sort(key=lambda t: -t[0])
            seen: set[frozenset] = set()
            new_beam: List[frozenset] = []
            for w2, C2, _ in scored_beam_candidates:
                if C2 in seen:
                    continue
                seen.add(C2)
                new_beam.append(C2)
                if len(new_beam) >= self.beam_width:
                    break
            beam = new_beam

        if global_best_C is None:
            return None

        sel = _indices_matching(global_best_C, Xd, E)
        if sel.size < self.min_rule_coverage:
            return None

        if len(global_best_C) == 0 and np.unique(y[sel]).size > 1:
            return None

        cls_final, _ = _majority_class(y[sel])
        return global_best_C, int(cls_final)

    def _sequential_cover(self, Xd: np.ndarray, y: np.ndarray, pool: np.ndarray) -> List[CN2Rule]:
        rules: List[CN2Rule] = []
        E = pool.copy()
        max_rules = 50

        while E.size >= self.min_rule_coverage and len(rules) < max_rules:
            out = self._find_best_rule(Xd, y, E)
            if out is None:
                break
            conj, cls_label = out
            mask = np.array([covers(conj, Xd[int(i)]) for i in E], dtype=bool)
            if not mask.any():
                break

            conds = tuple(sorted(conj))
            rules.append(CN2Rule(conditions=conds, predicted_class=int(cls_label)))
            E = E[~mask]
            if E.size == 0:
                break

        if E.size > 0:
            dc, _ = _majority_class(y[E])
        else:
            dc = self.default_class_

        rules.append(CN2Rule(conditions=tuple(), predicted_class=int(dc)))
        return rules

    def predict_one(self, x: np.ndarray) -> Tuple[int, int, CN2Rule]:
        xd = self.discretizer.transform(x.reshape(1, -1))[0].astype(np.int64)
        for rid, rule in enumerate(self.rules_):
            conj = frozenset(rule.conditions)
            if covers(conj, xd):
                return int(rule.predicted_class), rid, rule
        last = self.rules_[-1]
        return int(last.predicted_class), len(self.rules_) - 1, last

    def predict_one_with_confidence(self, x: np.ndarray) -> tuple[int, int, CN2Rule, float]:
        cls, rid, rule = self.predict_one(x)
        conf = 0.0
        if 0 <= rid < len(self.rule_stats_):
            conf = float(self.rule_stats_[rid].get("precision", 0.0))
        return cls, rid, rule, conf

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        out = np.empty(X.shape[0], dtype=np.int64)
        for i in range(X.shape[0]):
            cls, _, _ = self.predict_one(X[i])
            out[i] = cls
        return out

    def discretized_vector(self, x: np.ndarray) -> np.ndarray:
        return self.discretizer.transform(x.reshape(1, -1))[0].astype(np.int64)

    def format_rule_text(self, rule: CN2Rule) -> str:
        if not rule.conditions:
            cl = self.class_names[rule.predicted_class]
            return f"IF (else / default) THEN class = {cl}"

        parts = []
        for a, bv in sorted(rule.conditions):
            attr = self.feature_names[a]
            bi = int(bv)
            bl = BIN_LABELS[bi] if 0 <= bi < len(BIN_LABELS) else str(bi)
            parts.append(f"{attr} = {bl}")

        lhs = " AND ".join(parts)
        cl = self.class_names[rule.predicted_class]
        return f"IF {lhs} THEN class = {cl}"

    def export_rules_plain(self) -> List[str]:
        return [self.format_rule_text(r) for r in self.rules_]
