from __future__ import annotations

import html
import os
import shutil
import subprocess
from pathlib import Path

import pandas as pd

from src.generate_report_pdf import ReportTextPage, _display_model, build_report_text_pages
from src.utils import ensure_dir


REPO_URL = "https://github.com/Mengtaigu111/jiqixuexi"
ROOT = Path(__file__).resolve().parents[1]


def _asset_src(path: Path, html_path: Path) -> str:
    try:
        return html.escape(path.resolve().relative_to(html_path.parent.resolve()).as_posix())
    except ValueError:
        return html.escape(path.resolve().as_uri())


def _paragraphs(text: str) -> str:
    blocks: list[str] = []
    in_code = False
    code_lines: list[str] = []
    for raw in text.strip().splitlines():
        line = raw.rstrip()
        if line.strip() == "```text":
            in_code = True
            code_lines = []
            continue
        if line.strip() == "```" and in_code:
            in_code = False
            blocks.append("<pre><code>" + html.escape("\n".join(code_lines)) + "</code></pre>")
            continue
        if in_code:
            code_lines.append(line)
            continue
        if not line.strip():
            continue
        if line.startswith("[") and "]" in line[:5]:
            blocks.append(f"<p class=\"reference-line\">{html.escape(line)}</p>")
        else:
            blocks.append(f"<p>{html.escape(line)}</p>")
    if in_code and code_lines:
        blocks.append("<pre><code>" + html.escape("\n".join(code_lines)) + "</code></pre>")
    return "\n".join(blocks)


def _section_id(index: int) -> str:
    return f"sec-{index}"


def _summary_table(summary_path: Path, horizon: int) -> str:
    if not summary_path.exists():
        return "<p>缺少 summary.csv，无法生成结果表。</p>"
    summary = pd.read_csv(summary_path)
    subset = summary[summary["Horizon"].astype(int) == horizon].sort_values("MSE mean")
    rows = []
    for _, row in subset.iterrows():
        rows.append(
            "<tr>"
            f"<td>{html.escape(_display_model(str(row['Model'])))}</td>"
            f"<td>{float(row['MSE mean']):.2f} ± {float(row['MSE std']):.2f}</td>"
            f"<td>{float(row['MAE mean']):.2f} ± {float(row['MAE std']):.2f}</td>"
            f"<td>{int(row['Runs'])}</td>"
            "</tr>"
        )
    return (
        '<table class="booktabs">'
        f"<caption>表 {1 if horizon == 90 else 2}  {horizon} 天预测结果</caption>"
        "<thead><tr><th>模型</th><th>MSE mean ± std</th><th>MAE mean ± std</th><th>Runs</th></tr></thead>"
        "<tbody>"
        + "\n".join(rows)
        + "</tbody></table>"
    )


def _figure(src: Path, html_path: Path, label: str, caption: str) -> str:
    return (
        "<figure>"
        f"<img src=\"{_asset_src(src, html_path)}\" alt=\"{html.escape(caption)}\">"
        f"<figcaption class=\"figure-caption\"><strong>{html.escape(label)}</strong> {html.escape(caption)}</figcaption>"
        "</figure>"
    )


def _render_sections(pages: list[ReportTextPage], summary_path: Path, figures_dir: Path, html_path: Path) -> str:
    sections: list[str] = []
    for index, page in enumerate(pages[1:9], start=1):
        class_name = ' class="page-break-before"' if index in {4, 7} else ""
        sections.append(
            f"<section id=\"{_section_id(index)}\"{class_name}>"
            f"<h1>{html.escape(page.title)}</h1>"
            f"{_paragraphs(page.body)}"
            "</section>"
        )
    sections.append(
        '<section id="sec-results-table" class="page-break-before">'
        "<h1>3.2 结果汇总表</h1>"
        '<p>下表按预测长度拆分展示三个正式模型的原尺度 MSE/MAE mean ± std，所有组合均包含 5 个随机种子。</p>'
        + _summary_table(summary_path, 90)
        + _summary_table(summary_path, 365)
        + "</section>"
    )
    sections.append(
        '<section id="sec-figures" class="page-break-before">'
        "<h1>3.3 MSE 与 MAE 指标对比及预测曲线</h1>"
        '<div class="figure-grid">'
        + _figure(figures_dir / "metric_bar_mse.png", html_path, "图 1", "MSE 指标对比")
        + _figure(figures_dir / "metric_bar_mae.png", html_path, "图 2", "MAE 指标对比")
        + "</div>"
        + _figure(figures_dir / "prediction_comparison_90.png", html_path, "图 3", "90 天预测曲线对比")
        + _figure(figures_dir / "prediction_comparison_365.png", html_path, "图 4", "365 天预测曲线对比")
        + "</section>"
    )
    # Diagnostic text pages (§3.4 naive/capacity, §3.5 gap/ablation): indices 9-10.
    for offset, page in enumerate(pages[9:11], start=9):
        class_name = ' class="page-break-before"' if offset == 9 else ""
        sections.append(
            f"<section id=\"{_section_id(offset)}\"{class_name}>"
            f"<h1>{html.escape(page.title)}</h1>"
            f"{_paragraphs(page.body)}"
            "</section>"
        )
    sections.append(
        '<section id="sec-diag-figures" class="page-break-before">'
        "<h1>3.6 诊断分析配图</h1>"
        + _figure(figures_dir / "diag_naive_floor.png", html_path, "图 5", "深度模型 vs 朴素基线地板")
        + _figure(figures_dir / "diag_capacity_vs_error.png", html_path, "图 6", "参数量与测试误差")
        + _figure(figures_dir / "diag_overfitting_gap.png", html_path, "图 7", "过拟合诊断：验证/训练损失比")
        + _figure(figures_dir / "diag_ablation.png", html_path, "图 8", "DMSAFormer 逐模块消融")
        + "</section>"
    )
    # Discussion + references: indices 11-12.
    for offset, page in enumerate(pages[11:], start=11):
        class_name = ' class="page-break-before"' if offset in {11, 12} else ""
        sections.append(
            f"<section id=\"{_section_id(offset)}\"{class_name}>"
            f"<h1>{html.escape(page.title)}</h1>"
            f"{_paragraphs(page.body)}"
            "</section>"
        )
    return "\n".join(sections)


def build_report_html(report_dir: str | Path = "report", results_dir: str | Path = "results") -> str:
    report_dir = Path(report_dir)
    results_dir = Path(results_dir)
    html_path = report_dir / "report.html"
    summary_path = results_dir / "metrics" / "summary.csv"
    figures_dir = results_dir / "figures"
    font_path = report_dir / "report_assets" / "NotoSansCJKsc-Regular.otf"
    pages = build_report_text_pages(summary_path)
    cover = pages[0]
    sections = _render_sections(pages, summary_path, figures_dir, html_path)

    toc_items = [
        ("sec-1", "1. 问题介绍"),
        ("sec-2", "1.1 数据预处理与样本构造"),
        ("sec-3", "1.2 实验设置与泄漏控制"),
        ("sec-4", "2. 基线模型方法"),
        ("sec-5", "2.1 DMSAFormer 方法细节"),
        ("sec-6", "2.2 公平对比口径与校准消融"),
        ("sec-7", "3. 结果与分析"),
        ("sec-results-table", "3.2 结果汇总表"),
        ("sec-figures", "3.3 指标图与预测曲线"),
        ("sec-9", "3.4 诊断分析一：朴素基线地板与参数量代价"),
        ("sec-10", "3.5 诊断分析二：过拟合缺口与逐模块消融"),
        ("sec-diag-figures", "3.6 诊断图表"),
        ("sec-11", "4. 讨论与后续工作"),
        ("sec-12", "参考文献"),
    ]
    toc = "\n".join(f"<li><a href=\"#{target}\">{html.escape(label)}</a></li>" for target, label in toc_items)

    font_face = ""
    if font_path.exists():
        font_face = (
            "@font-face { font-family: 'ReportCJK'; "
            f"src: url('{html.escape(font_path.resolve().as_uri())}') format('opentype'); }}"
        )

    css = f"""
{font_face}
@page {{
  size: A4;
  margin: 17mm 18mm 18mm 18mm;
}}
@page :first {{
  margin: 0;
}}
* {{
  box-sizing: border-box;
}}
body {{
  margin: 0;
  color: #2f3437;
  background: #ffffff;
  font-family: ReportCJK, "Noto Sans CJK SC", "Microsoft YaHei", Arial, sans-serif;
  font-size: 10.5pt;
  line-height: 1.62;
  text-align: justify;
  text-align-last: left;
}}
.cover {{
  width: 210mm;
  height: 297mm;
  position: relative;
  overflow: hidden;
  page-break-after: always;
  background:
    linear-gradient(90deg, rgba(31,78,121,0.10) 0 18%, transparent 18% 100%),
    linear-gradient(0deg, rgba(91,141,163,0.08) 0 12%, transparent 12% 100%),
    #fbfcfd;
}}
.cover::before {{
  content: "";
  position: absolute;
  left: 18mm;
  top: 18mm;
  right: 18mm;
  bottom: 18mm;
  border: 1.4pt solid #17365d;
}}
.cover::after {{
  content: "";
  position: absolute;
  left: 28mm;
  top: 28mm;
  width: 58mm;
  height: 4mm;
  background: #1f4e79;
}}
.cover-content {{
  position: absolute;
  left: 34mm;
  right: 26mm;
  top: 58mm;
}}
.cover-kicker {{
  font-size: 10pt;
  letter-spacing: 0;
  color: #1f4e79;
  font-weight: 700;
  border-bottom: 1pt solid #a7bac9;
  padding-bottom: 7mm;
}}
.cover h1 {{
  margin: 18mm 0 7mm 0;
  color: #17365d;
  font-size: 27pt;
  line-height: 1.22;
  font-weight: 800;
  text-align: left;
}}
.cover-subtitle {{
  color: #4f626f;
  font-size: 13pt;
  line-height: 1.5;
  margin-bottom: 22mm;
}}
.cover-meta {{
  border-top: 1pt solid #b8c7d2;
  padding-top: 8mm;
  color: #3f4e58;
  font-size: 10.5pt;
  line-height: 1.9;
}}
.toc-page {{
  page-break-after: always;
  padding: 18mm 20mm;
}}
.toc-page h1 {{
  color: #17365d;
  font-size: 22pt;
  margin: 0 0 12mm 0;
  border-bottom: 1.2pt solid #17365d;
  padding-bottom: 5mm;
}}
.toc-page ol {{
  margin: 0;
  padding-left: 0;
  list-style: none;
}}
.toc-page li {{
  border-bottom: 0.5pt solid #d6dee4;
  padding: 3.4mm 0;
  font-size: 11pt;
}}
.toc-page a {{
  color: #2f3437;
  text-decoration: none;
}}
.running-header {{
  height: 0;
  overflow: hidden;
}}
.page-footer {{
  height: 0;
  overflow: hidden;
}}
.report-shell {{
  padding: 0;
}}
section {{
  margin: 0 0 7mm 0;
  page-break-inside: auto;
}}
.page-break-before {{
  break-before: page;
  page-break-before: always;
}}
h1 {{
  color: #17365d;
  font-size: 16.5pt;
  line-height: 1.25;
  margin: 0 0 4.8mm 0;
  padding: 0 0 2.3mm 0;
  border-bottom: 1pt solid #8aa4b8;
  page-break-after: avoid;
}}
p {{
  margin: 0 0 3.2mm 0;
}}
pre {{
  max-width: 100%;
  white-space: pre-wrap;
  word-wrap: break-word;
  background: #f4f6f8;
  border-left: 3pt solid #1f4e79;
  padding: 3.5mm 4mm;
  margin: 3.5mm 0;
  font-size: 8.6pt;
  line-height: 1.42;
  color: #25313a;
}}
code {{
  font-family: "Consolas", "Courier New", monospace;
}}
.booktabs {{
  width: 100%;
  max-width: 100%;
  border-collapse: collapse;
  margin: 5mm 0 7mm 0;
  font-size: 9.4pt;
}}
.booktabs caption {{
  caption-side: top;
  text-align: left;
  font-weight: 700;
  color: #17365d;
  margin-bottom: 2mm;
}}
.booktabs thead {{
  border-top: 1.5pt solid #17365d;
  border-bottom: 1pt solid #17365d;
}}
.booktabs tbody {{
  border-bottom: 1.3pt solid #17365d;
}}
.booktabs th,
.booktabs td {{
  padding: 2.4mm 2mm;
  text-align: center;
  vertical-align: middle;
  border-left: none;
  border-right: none;
}}
.booktabs tbody tr:nth-child(even) {{
  background: #f5f8fa;
}}
figure {{
  margin: 5mm 0 8mm 0;
  page-break-inside: avoid;
  text-align: center;
}}
figure img {{
  max-width: 100%;
  max-height: 118mm;
  height: auto;
}}
.figure-grid {{
  display: grid;
  grid-template-columns: 1fr;
  gap: 4mm;
}}
.figure-grid figure img {{
  max-height: 86mm;
}}
.figure-caption {{
  margin-top: 2mm;
  color: #4f626f;
  font-size: 9pt;
  text-align: center;
  text-align-last: center;
}}
.reference-line {{
  padding-left: 7mm;
  text-indent: -7mm;
  font-size: 9.3pt;
  line-height: 1.48;
}}
"""

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>基于深度学习的家庭电力消耗多变量时间序列预测</title>
  <style>{css}</style>
</head>
<body>
  <div class="running-header">基于深度学习的家庭电力消耗多变量时间序列预测</div>
  <div class="page-footer">2026 年专硕机器学习课程项目报告</div>
  <section class="cover">
    <div class="cover-content">
      <div class="cover-kicker">2026 年专硕机器学习课程项目报告</div>
      <h1>{html.escape(cover.title)}</h1>
      <div class="cover-subtitle">LSTM、Transformer 与 DMSAFormer 的三模型公平比较研究</div>
      <div class="cover-meta">
        <div>GitHub：{html.escape(REPO_URL)}</div>
        <div>正式实验：三模型 x 两个预测长度 x 五个 seed，共 30 次正式对比实验</div>
        <div>作者贡献：独立完成数据预处理、模型实现、实验设计、结果分析与报告撰写</div>
        <div>研究领域：机器学习与时间序列预测</div>
        <div>工具说明：AI 工具辅助报告撰写与格式整理，实验结果来自本项目真实代码运行</div>
      </div>
    </div>
  </section>
  <section class="toc-page">
    <h1>目录</h1>
    <ol>{toc}</ol>
  </section>
  <main class="report-shell">
    {sections}
  </main>
</body>
</html>
"""


def write_report_html(report_dir: str | Path = "report", results_dir: str | Path = "results") -> Path:
    report_dir = ensure_dir(report_dir)
    html_path = report_dir / "report.html"
    html_path.write_text(build_report_html(report_dir=report_dir, results_dir=results_dir), encoding="utf-8")
    return html_path


def html_to_pdf_command(html_path: str | Path, pdf_path: str | Path) -> list[str]:
    script_path = ROOT / "scripts" / "html_to_pdf.js"
    return ["node", str(script_path), str(Path(html_path)), str(Path(pdf_path))]


def convert_html_to_pdf(html_path: str | Path, pdf_path: str | Path) -> Path:
    html_path = Path(html_path)
    pdf_path = Path(pdf_path)
    env = os.environ.copy()
    conda_prefix = env.get("CONDA_PREFIX")
    if conda_prefix:
        conda_lib = str(Path(conda_prefix) / "lib")
        existing = env.get("LD_LIBRARY_PATH")
        env["LD_LIBRARY_PATH"] = conda_lib if not existing else f"{conda_lib}:{existing}"
    subprocess.run(html_to_pdf_command(html_path, pdf_path), cwd=ROOT, env=env, check=True)
    return pdf_path


def generate_pdf(report_dir: str | Path = "report", results_dir: str | Path = "results") -> Path:
    report_dir = ensure_dir(report_dir)
    html_path = write_report_html(report_dir=report_dir, results_dir=results_dir)
    out_path = report_dir / "ML_household_power_report.pdf"
    convert_html_to_pdf(html_path, out_path)
    shutil.copyfile(out_path, report_dir / "report.pdf")
    return out_path


def main() -> None:
    path = generate_pdf()
    print(f"Saved HTML/PDF report: {path}")


if __name__ == "__main__":
    main()
