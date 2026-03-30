from __future__ import annotations

import csv
import json
from pathlib import Path


def _load_metrics(metrics_path: Path) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    if not metrics_path.exists():
        return metrics

    with metrics_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = str(row.get("model", "")).strip()
            if not name:
                continue
            metrics[name] = {
                "fid": float(row.get("fid", 0.0)),
                "train_time": float(row.get("train_time_min", 0.0)),
                "params": float(row.get("params_m", 0.0)),
            }
    return metrics


def _load_run_summary(summary_path: Path) -> dict:
    if not summary_path.exists():
        return {}
    with summary_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _fmt(v: float, nd: int = 4) -> str:
    return f"{v:.{nd}f}"


def _latex_escape(text: str) -> str:
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
        .replace("#", r"\#")
        .replace("$", r"\$")
    )


def generate_latex_report(root_dir: Path) -> None:
    metrics = _load_metrics(root_dir / "outputs" / "comparison_metrics.csv")
    summary = _load_run_summary(root_dir / "outputs" / "run_summary.json")

    cfg = summary.get("config", {}) if isinstance(summary, dict) else {}
    profile = str(summary.get("profile", "unknown")) if isinstance(summary, dict) else "unknown"
    quick_test = bool(summary.get("quick_test", False)) if isinstance(summary, dict) else False
    world_size = int(summary.get("world_size", 1)) if isinstance(summary, dict) else 1

    image_size = int(cfg.get("image_size", 96))
    epochs_diff = int(cfg.get("epochs_diffusion", 0))
    epochs_flow = int(cfg.get("epochs_flow", 0))
    epochs_ar = int(cfg.get("epochs_autoregressive", 0))
    fid_num_gen = int(cfg.get("fid_num_gen", 0))
    diff_steps = int(cfg.get("diffusion_sample_steps", 0))
    flow_steps = int(cfg.get("flow_sample_steps", 0))
    ar_refine = int(cfg.get("ar_refine_steps", 0))
    max_train_samples = cfg.get("max_train_samples", None)

    ranking = sorted(metrics.items(), key=lambda kv: kv[1].get("fid", 1e9))
    best_model = ranking[0][0] if ranking else "N/A"

    qt_note = "是" if quick_test else "否"
    report_mode = "快速冒烟报告" if quick_test else "完整训练报告"

    if quick_test:
        conclusion_line = (
            "本次运行为 quick-test 冒烟配置，结果主要用于验证训练和采样链路是否正常，"
            "不宜用于对三类模型做最终性能结论。"
        )
    else:
        conclusion_line = (
            f"在本次配置下，FID 最优模型为 {best_model}。建议结合可视化与训练成本共同判断模型选型。"
        )

    train_data_desc = "全量可用训练集"
    if max_train_samples is not None:
        train_data_desc = f"子集训练（max_train_samples={max_train_samples}）"

    profile_tex = _latex_escape(profile)
    train_data_desc_tex = _latex_escape(train_data_desc)
    conclusion_line_tex = _latex_escape(conclusion_line)

    latex = f"""\\documentclass[UTF8,a4paper,12pt]{{ctexart}}
\\usepackage{{geometry}}
\\usepackage{{graphicx}}
\\usepackage{{booktabs}}
\\usepackage{{amsmath,amssymb}}
\\usepackage{{caption}}
\\usepackage{{subcaption}}
\\usepackage{{float}}

\\geometry{{left=2.2cm,right=2.2cm,top=2.3cm,bottom=2.3cm}}
\\setlength{{\\parindent}}{{2em}}
\\setlength{{\\parskip}}{{0.35em}}
\\renewcommand{{\\arraystretch}}{{1.15}}

\\title{{机器学习实验报告\\\\实验八：生成模型对比（{report_mode}）}}
\\author{{实验人员: 柯力洲}}
\\date{{\\today}}

\\begin{{document}}
\\maketitle

\\section*{{一、实验目的}}
对比 Diffusion、Flow Matching、Autoregressive 三种生成范式在同一训练脚本下的表现，
输出可复现指标与可视化结果，并基于真实运行配置生成报告。

\\section*{{二、真实运行配置}}
\\begin{{itemize}}
    \\item 运行 profile: \\texttt{{{profile_tex}}}
    \\item quick-test: {qt_note}
    \\item GPU 并行规模 (world\_size): {world_size}
    \\item 输入分辨率: ${image_size}\\times{image_size}$
    \\item 训练轮数: Diffusion={epochs_diff}, Flow={epochs_flow}, AR={epochs_ar}
    \\item 采样设置: Diffusion steps={diff_steps}, Flow steps={flow_steps}, AR refine={ar_refine}
    \\item FID 生成样本数: {fid_num_gen}
    \\item 训练数据使用: {train_data_desc_tex}
\\end{{itemize}}

\\section*{{三、定量结果}}
\\begin{{table}}[H]
\\centering
\\caption{{三类模型的 FID、训练耗时与参数规模（FID 越低越好）}}
\\begin{{tabular}}{{lccc}}
\\toprule
模型 & FID $\\downarrow$ & 训练耗时(min) & 参数量(M) \\\\
\\midrule
Diffusion & {_fmt(metrics.get("diffusion", {}).get("fid", 0.0), 4)} & {_fmt(metrics.get("diffusion", {}).get("train_time", 0.0), 2)} & {_fmt(metrics.get("diffusion", {}).get("params", 0.0), 3)} \\\\
Flow Matching & {_fmt(metrics.get("flow_matching", {}).get("fid", 0.0), 4)} & {_fmt(metrics.get("flow_matching", {}).get("train_time", 0.0), 2)} & {_fmt(metrics.get("flow_matching", {}).get("params", 0.0), 3)} \\\\
Autoregressive & {_fmt(metrics.get("autoregressive", {}).get("fid", 0.0), 4)} & {_fmt(metrics.get("autoregressive", {}).get("train_time", 0.0), 2)} & {_fmt(metrics.get("autoregressive", {}).get("params", 0.0), 3)} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}

\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.72\\textwidth]{{figures/fid_barplot.png}}
\\caption{{FID 对比柱状图。}}
\\end{{figure}}

\\section*{{四、可视化结果}}
\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.95\\textwidth]{{figures/class_panels_cat_dog_car_hq.png}}
\\caption{{按类别（cat/dog/automobile）对比三种模型的生成结果。}}
\\end{{figure}}

\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.95\\textwidth]{{figures/ultra_curated_comparison.png}}
\\caption{{三种模型的样本画廊对比。}}
\\end{{figure}}

\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.95\\textwidth]{{figures/readability_zoom_panels.png}}
\\caption{{中心区域放大对比，用于观察纹理与可读性。}}
\\end{{figure}}

\\section*{{五、结论}}
{conclusion_line_tex}

\\end{{document}}
"""

    with (root_dir / "report8.tex").open("w", encoding="utf-8") as f:
        f.write(latex)


if __name__ == "__main__":
    generate_latex_report(Path(__file__).parent.parent)
