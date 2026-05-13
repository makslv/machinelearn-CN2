from __future__ import annotations

import base64
import io
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_WEB = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.append(str(_ROOT))
if str(_WEB) not in sys.path:
    sys.path.insert(0, str(_WEB))

import numpy as np
import matplotlib
from flask import Flask, render_template, request, jsonify, send_file
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, plot_tree

matplotlib.use("Agg")

from src.cn2 import BIN_LABELS
from src.decision_tree import describe_decision_path, export_tree_ascii

from model_loader import (
    get_classifier,
    get_decision_tree,
    get_fit_meta,
    get_train_test,
    retrain,
)

app = Flask(__name__)


def _build_instance_figure(
    x: np.ndarray,
    pred_class: int,
    *,
    chart_title: str = "Prediction vs class mean (training subset)",
) -> go.Figure:
    clf = get_classifier()
    names_short = clf.feature_names_short
    class_names = clf.class_names
    Xtr = clf.X_train_
    ytr = clf.y_train_

    centroid = Xtr[ytr == pred_class].mean(axis=0) if np.any(ytr == pred_class) else Xtr.mean(axis=0)

    fig = go.Figure(
        data=[
            go.Bar(name="Your instance", x=names_short, y=x.tolist(), marker_color="rgba(176,38,255,.9)"),
            go.Bar(
                name=f"Mean ({class_names[pred_class]})",
                x=names_short,
                y=centroid.tolist(),
                marker_color="rgba(0,255,224,.75)",
            ),
        ]
    )

    fig.update_layout(
        template="plotly_dark",
        autosize=True,
        margin=dict(l=44, r=140, t=42, b=48),
        barmode="group",
        bargap=0.2,
        title=dict(
            text=chart_title,
            font=dict(size=12),
            y=0.97,
            yanchor="top",
            x=0,
            xanchor="left",
        ),
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            x=1.01,
            xanchor="left",
            font=dict(size=11),
        ),
        xaxis=dict(tickangle=0, automargin=True, title=""),
        yaxis=dict(title="cm", automargin=True, title_standoff=6),
    )
    return fig


def _tree_plot_base64(
    tree: DecisionTreeClassifier,
    feature_names: list[str],
    class_names: list[str],
    *,
    x: np.ndarray | None = None,
) -> str:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5), facecolor="#0b0b0f")
    ax.set_facecolor("#0b0b0f")
    artists = plot_tree(
        tree,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        impurity=False,
        fontsize=8,
        ax=ax,
    )
    if x is not None:
        x_row = np.asarray(x, dtype=np.float64).reshape(1, -1)
        node_ids = tree.decision_path(x_row).indices.tolist()
        node_set = set(int(n) for n in node_ids)
        for i, artist in enumerate(artists):
            bbox = artist.get_bbox_patch() if hasattr(artist, "get_bbox_patch") else None
            if bbox is None:
                continue
            if i in node_set:
                bbox.set_facecolor("#00ffd0")
                bbox.set_edgecolor("#ffffff")
                bbox.set_linewidth(2.4)
                bbox.set_alpha(0.96)
            else:
                bbox.set_alpha(0.35)
    else:
        for artist in artists:
            bbox = artist.get_bbox_patch() if hasattr(artist, "get_bbox_patch") else None
            if bbox is not None:
                bbox.set_alpha(0.8)
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=180)
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("ascii")


def _parse_features(data: dict) -> tuple[np.ndarray | None, tuple[dict, int] | None]:
    feats = data.get("features")
    if not isinstance(feats, list) or len(feats) != 4:
        return None, (
            {
                "error": "Provide exactly four numeric features (4D vector).",
            },
            400,
        )
    try:
        x = np.array([float(f) for f in feats], dtype=np.float64)
    except (TypeError, ValueError):
        return None, ({"error": "Feature values must be valid numbers."}, 400)
    if not np.all(np.isfinite(x)):
        return None, ({"error": "Feature values must be finite numbers."}, 400)
    return x, None


def _cn2_payload(clf, x: np.ndarray) -> dict:
    cls_idx, rule_id, rule_obj, conf = clf.predict_one_with_confidence(x)
    disc = clf.discretized_vector(x)
    rule_text = clf.format_rule_text(rule_obj)
    disc_labels = []
    for a, bi in enumerate(disc):
        bi_i = int(bi)
        lbl = BIN_LABELS[bi_i] if 0 <= bi_i < len(BIN_LABELS) else str(bi_i)
        disc_labels.append({"attr": clf.feature_names[a], "bin": bi_i, "label": lbl})
    stats = {}
    if 0 <= rule_id < len(clf.rule_stats_):
        stats = clf.rule_stats_[rule_id]
    fig = _build_instance_figure(x, cls_idx, chart_title="CN2 — instance vs class mean (training subset)")
    return {
        "model": "cn2",
        "class_index": cls_idx,
        "class_name": clf.class_names[cls_idx],
        "confidence": float(conf),
        "rule_index": rule_id,
        "rule_text": rule_text,
        "rule_coverage": int(stats.get("coverage", 0)),
        "rule_precision": float(stats.get("precision", 0.0)),
        "discretized": disc.tolist(),
        "discretized_labels": disc_labels,
        "figure": pio.to_json(fig),
    }


def _tree_payload(clf, dt, x: np.ndarray) -> dict:
    probs = dt.predict_proba(x.reshape(1, -1))[0]
    cls_idx = int(np.argmax(probs))
    path_text = describe_decision_path(dt, clf.feature_names_short, clf.class_names, x)
    fi = {clf.feature_names_short[i]: float(dt.feature_importances_[i]) for i in range(len(clf.feature_names_short))}
    fig = _build_instance_figure(x, cls_idx, chart_title="Decision tree — instance vs class mean (training subset)")
    return {
        "model": "tree",
        "class_index": cls_idx,
        "class_name": clf.class_names[cls_idx],
        "confidence": float(np.max(probs)),
        "rule_index": None,
        "rule_text": path_text,
        "discretized": [],
        "discretized_labels": [],
        "tree_feature_importances": fi,
        "tree_plot_highlight": _tree_plot_base64(dt, clf.feature_names_short, clf.class_names, x=x),
        "figure": pio.to_json(fig),
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/meta", methods=["GET"])
def api_meta():
    clf = get_classifier()
    meta = get_fit_meta()
    dt = get_decision_tree()

    rules = [{"id": i, "text": clf.format_rule_text(rule)} for i, rule in enumerate(clf.rules_)]
    edges = [[float(v) for v in e] for e in clf.edges_]
    tree_export = ""
    tree_fi: dict[str, float] = {}
    tree_plot = ""
    meta_warnings: list[str] = []
    try:
        tree_export = export_tree_ascii(dt, clf.feature_names_short)
    except Exception as e:  # pragma: no cover - defensive API hardening
        meta_warnings.append(f"tree_export: {e}")
    try:
        tree_fi = {
            clf.feature_names_short[i]: float(dt.feature_importances_[i])
            for i in range(len(clf.feature_names_short))
        }
    except Exception as e:  # pragma: no cover
        meta_warnings.append(f"tree_feature_importances: {e}")
    try:
        tree_plot = _tree_plot_base64(dt, clf.feature_names_short, clf.class_names)
    except Exception as e:  # pragma: no cover
        meta_warnings.append(f"tree_plot: {e}")

    return jsonify(
        {
            "feature_names": clf.feature_names,
            "class_names": clf.class_names,
            "bin_labels": BIN_LABELS,
            "n_bins": clf.n_bins,
            "rules": rules,
            "edges": edges,
            "models": ["cn2", "tree"],
            "datasets": ["iris", "weather", "loan"],
            "tree_export_text": tree_export,
            "tree_feature_importances": tree_fi,
            "tree_plot": tree_plot,
            "tree_plot_url": "/api/tree-plot.png",
            "meta_warnings": meta_warnings,
            **meta,
        }
    )


@app.route("/api/tree-plot.png", methods=["GET"])
def api_tree_plot_png():
    clf = get_classifier()
    dt = get_decision_tree()
    data_url = _tree_plot_base64(dt, clf.feature_names_short, clf.class_names)
    if "," not in data_url:
        return jsonify({"error": "Could not build tree image"}), 500
    b64 = data_url.split(",", 1)[1]
    png_bytes = base64.b64decode(b64)
    return send_file(io.BytesIO(png_bytes), mimetype="image/png")


@app.route("/api/classify", methods=["POST"])
def api_classify():
    data = request.json or {}
    x, err = _parse_features(data)
    if err is not None:
        return jsonify(err[0]), err[1]
    assert x is not None

    clf = get_classifier()
    dt = get_decision_tree()
    model_kind = str(data.get("model") or "cn2").strip().lower()
    if model_kind not in ("cn2", "tree"):
        return jsonify({"error": 'Field "model" must be "cn2" or "tree".'}), 400

    if model_kind == "tree":
        return jsonify(_tree_payload(clf, dt, x))
    return jsonify(_cn2_payload(clf, x))


@app.route("/api/compare", methods=["POST"])
def api_compare():
    data = request.json or {}
    x, err = _parse_features(data)
    if err is not None:
        return jsonify(err[0]), err[1]
    assert x is not None
    clf = get_classifier()
    dt = get_decision_tree()
    cn2_row = _cn2_payload(clf, x)
    tree_row = _tree_payload(clf, dt, x)
    disagree = cn2_row["class_index"] != tree_row["class_index"]
    return jsonify(
        {
            "cn2": cn2_row,
            "tree": tree_row,
            "disagree": bool(disagree),
            "note": "Predictions differ: inspect rule/path and confidence."
            if disagree
            else "Both models agree on this input.",
        }
    )


@app.route("/api/train", methods=["POST"])
def api_train():
    data = request.json or {}
    dataset = str(data.get("dataset") or "iris").strip().lower()
    random_state = int(data.get("random_state", 42))
    test_size = float(data.get("test_size", 0.25))
    cn2_params = {
        "n_bins": int(data.get("cn2_n_bins", 3)),
        "beam_width": int(data.get("cn2_beam_width", 6)),
        "max_conditions": int(data.get("cn2_max_conditions", 5)),
        "min_rule_coverage": int(data.get("cn2_min_rule_coverage", 4)),
    }
    tree_params = {
        "max_depth": int(data.get("tree_max_depth", 4)),
        "min_samples_leaf": int(data.get("tree_min_samples_leaf", 2)),
    }
    if not 0.1 <= test_size <= 0.6:
        return jsonify({"error": "test_size must be between 0.1 and 0.6"}), 400
    try:
        st = retrain(
            dataset=dataset,
            test_size=test_size,
            random_state=random_state,
            cn2_params=cn2_params,
            tree_params=tree_params,
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    clf = st["classifier"]
    dt = st["tree"]
    meta = st["meta"]
    depths = list(range(1, max(2, int(tree_params["max_depth"]) + 1)))
    X_train, X_test, y_train, y_test = st["X_train"], st["X_test"], st["y_train"], st["y_test"]
    overfit = []
    for d in depths:
        mdl = DecisionTreeClassifier(
            criterion="entropy",
            max_depth=int(d),
            min_samples_leaf=int(tree_params["min_samples_leaf"]),
            random_state=random_state,
        ).fit(X_train, y_train)
        overfit.append(
            {
                "max_depth": int(d),
                "train_accuracy": float(accuracy_score(y_train, mdl.predict(X_train))),
                "test_accuracy": float(accuracy_score(y_test, mdl.predict(X_test))),
            }
        )
    return jsonify(
        {
            "meta": meta,
            "cn2_rule_count": len(clf.rules_),
            "tree_depth": int(dt.get_depth()),
            "tree_export_text": export_tree_ascii(dt, clf.feature_names_short),
            "tree_plot": _tree_plot_base64(dt, clf.feature_names_short, clf.class_names),
            "overfitting_curve": overfit,
        }
    )


@app.route("/api/predict", methods=["POST"])
def api_predict_legacy():
    return (
        jsonify(
            {
                "error": 'Deprecated. Use POST /api/classify with {"features":[...], "model":"cn2"|"tree"}.',
            }
        ),
        410,
    )


if __name__ == "__main__":
    import os

    port = int(os.environ.get("PORT", "5050"))
    print(f"\n  CN2 + Decision tree demo → http://127.0.0.1:{port}/\n")
    app.run(debug=True, host="127.0.0.1", port=port)
