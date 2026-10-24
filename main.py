"""一键运行完整流程:  python main.py
跳过下载 (数据已存在时):  python main.py --skip-download

这个文件已经写好。等 src/ 里的函数都实现了，它就能跑通。
"""
import argparse
import argparse

import pandas as pd

from src.ablation import bootstrap_ci, crp_ablation
...
from src.ablation import bootstrap_ci, crp_ablation

from src.download import download_all
from src.evaluate import evaluate_all, plot_curves
from src.explain import shap_summary
from src.preprocess import run_preprocessing
from src.train import (build_preprocessor, choose_thresholds, cross_validate_models,
                       fit_all, get_models, split_data)
from src.utils import ensure_dir, load_config, set_seed


def main(config_path: str, skip_download: bool) -> None:
    cfg = load_config(config_path)
    set_seed(cfg["seed"])
    figures = ensure_dir(cfg["figures_dir"])
    results = ensure_dir(cfg["results_dir"])

    # 1. 下载
    if not skip_download:
        print("== Step 1/5: Downloading NHANES files ==")
        download_all(cfg)

    # 2. 预处理
    print("== Step 2/5: Preprocessing ==")
    df = run_preprocessing(cfg)
    numeric = cfg["features"]["numeric"]
    categorical = cfg["features"]["categorical"]
    target = cfg["target"]

    X_train, X_test, y_train, y_test = split_data(
        df, numeric + categorical, target, cfg["test_size"], cfg["seed"])
    print(f"Train n={len(y_train)} (prevalence {y_train.mean():.1%}), "
          f"Test n={len(y_test)}")

    # 3. 交叉验证
    print("== Step 3/5: Cross-validation ==")
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    models = get_models(cfg["seed"], pos_weight)
    preprocessor = build_preprocessor(numeric, categorical)
    cv_table = cross_validate_models(models, preprocessor, X_train, y_train,
                                     cfg["cv_folds"], cfg["seed"])
    cv_table.to_csv(results / "cv_results.csv", index=False)
    print(cv_table.round(3).to_string(index=False))

    # 4. 测试集评估
    print("== Step 4/5: Test-set evaluation ==")
    fitted = fit_all(models, preprocessor, X_train, y_train)
    thresholds = choose_thresholds(models, preprocessor, X_train, y_train,
                                   cfg["cv_folds"], cfg["seed"])
    print("Youden thresholds (chosen on training OOF):",
          {k: round(v, 3) for k, v in thresholds.items()})
    test_table = evaluate_all(fitted, X_test, y_test, thresholds)
    test_table.to_csv(results / "test_results.csv", index=False)
    print(test_table.round(3).to_string(index=False))
    plot_curves(fitted, X_test, y_test, str(figures))
    ci_rows = []
    for name, pipe in fitted.items():
        auc, lo, hi = bootstrap_ci(y_test, pipe.predict_proba(X_test)[:, 1],
                                   seed=cfg["seed"])
        ci_rows.append({"model": name, "auroc": auc, "ci_low": lo, "ci_high": hi})
    ci_table = pd.DataFrame(ci_rows)
    ci_table.to_csv(results / "test_auroc_ci.csv", index=False)
    print("Test AUROC with 95% bootstrap CI:")
    print(ci_table.round(3).to_string(index=False))

    # 5. 可解释性
    print("== Step 5/5: SHAP ==")
    shap_summary(fitted["XGBoost"], X_test, str(figures / "shap_summary.png"))
    shap_summary(fitted["LogisticRegression"], X_test,
                 str(figures / "shap_summary_lr.png"))
    print("== Extra: hs-CRP ablation ==")
    abl = crp_ablation(models, numeric, categorical, X_train, y_train,
                       X_test, y_test, seed=cfg["seed"])
    abl.to_csv(results / "crp_ablation.csv", index=False)
    print(abl.round(3).to_string(index=False))
    print("Done. See results/ and figures/.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NHANES depression risk prediction")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()
    main(args.config, args.skip_download)
