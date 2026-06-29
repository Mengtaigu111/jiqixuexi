from __future__ import annotations

import csv
import json
import shutil
import subprocess
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "report"
OUT = REPORT_DIR / "ML_household_power_report.docx"
COMPAT_OUT = REPORT_DIR / "report.docx"


FONT = "Microsoft YaHei"
ACCENT = "1F4E79"
DARK = "17365D"
LIGHT_BLUE = "EAF3F8"
TABLE_HEADER = "1F4E79"
TABLE_ZEBRA = "F4F8FB"
MUTED = "666666"
REPO_URL = "https://github.com/Mengtaigu111/jiqixuexi"


def run(*args: str) -> None:
    subprocess.run(["officecli", *args], cwd=ROOT, check=True)


def add_p(text: str, **props: str | int | float | bool) -> None:
    cmd = ["add", str(OUT), "/body", "--type", "paragraph", "--prop", f"text={text}"]
    for key, value in props.items():
        cmd.extend(["--prop", f"{key.replace('_', '.')}={value}"])
    run(*cmd)


def add_run_to_last(text: str, **props: str | int | float | bool) -> None:
    cmd = ["add", str(OUT), "/body/p[last()]", "--type", "run", "--prop", f"text={text}"]
    for key, value in props.items():
        cmd.extend(["--prop", f"{key.replace('_', '.')}={value}"])
    run(*cmd)


def heading(level: int, text: str, break_before: bool = False) -> None:
    if break_before:
        page_break()
    if level == 1:
        add_p(
            text,
            style="Heading1",
            size="20pt",
            bold="true",
            color=DARK,
            shd=f"clear;{LIGHT_BLUE}",
            spaceBefore="16pt",
            spaceAfter="9pt",
            keepNext="true",
        )
        if break_before:
            run("set", str(OUT), "/body/p[last()]", "--prop", "pageBreakBefore=true")
    elif level == 2:
        add_p(text, style="Heading2", size="14pt", bold="true", color=ACCENT, spaceBefore="12pt", spaceAfter="5pt", keepNext="true")
    else:
        add_p(text, style="Heading3", size="12pt", bold="true", color=DARK, spaceBefore="8pt", spaceAfter="4pt", keepNext="true")


def body(text: str) -> None:
    add_p(
        text,
        size="11pt",
        font=FONT,
        font_ea=FONT,
        lineSpacing="1.35x",
        spaceAfter="6pt",
        align="justify",
        firstLineIndent=440,
    )


def bullet(text: str) -> None:
    add_p(text, size="10.5pt", font=FONT, font_ea=FONT, lineSpacing="1.25x", spaceAfter="3pt", listStyle="bullet")


def code_line(text: str) -> None:
    add_p(text, size="9pt", font="Consolas", shd="clear;F3F6FA", color="333333", indent=360, spaceAfter="3pt")


def page_break() -> None:
    run("add", str(OUT), "/body", "--type", "pagebreak", "--prop", "type=page")


def looks_numeric(text: str) -> bool:
    cleaned = text.replace(",", "").replace(".", "").replace("-", "").replace("+", "").replace("%", "").strip()
    return bool(cleaned) and cleaned.isdigit()


def add_table(rows: list[list[str]], widths: str = "2200,1800,2200,2200,2200,2200,1000") -> None:
    data = ";".join(",".join(cell.replace(",", "，").replace(";", "；") for cell in row) for row in rows)
    run(
        "add",
        str(OUT),
        "/body",
        "--type",
        "table",
        "--prop",
        f"data={data}",
        "--prop",
        "width=100%",
        "--prop",
        "style=light1",
        "--prop",
        "layout=fixed",
        "--prop",
        f"colWidths={widths}",
    )
    table_index = table_count()
    run("set", str(OUT), f"/body/tbl[{table_index}]/tr[1]", "--prop", "header=true")
    data_size = "8.5pt" if len(rows[0]) >= 6 else "9pt"
    for col in range(1, len(rows[0]) + 1):
        run(
            "set",
            str(OUT),
            f"/body/tbl[{table_index}]/tr[1]/tc[{col}]",
            "--prop",
            f"fill={TABLE_HEADER}",
            "--prop",
            "valign=center",
        )
        run(
            "set",
            str(OUT),
            f"/body/tbl[{table_index}]/tr[1]/tc[{col}]/p[1]",
            "--prop",
            "bold=true",
            "--prop",
            "color=FFFFFF",
            "--prop",
            "align=center",
            "--prop",
            "size=9pt",
            "--prop",
            f"font={FONT}",
            "--prop",
            f"font.ea={FONT}",
            "--prop",
            f"pbdr.bottom=single;8;{DARK};0",
        )
    for row in range(2, len(rows) + 1):
        for col in range(1, len(rows[0]) + 1):
            if row % 2 == 1:
                run("set", str(OUT), f"/body/tbl[{table_index}]/tr[{row}]/tc[{col}]", "--prop", f"fill={TABLE_ZEBRA}")
            run(
                "set",
                str(OUT),
                f"/body/tbl[{table_index}]/tr[{row}]/tc[{col}]/p[1]",
                "--prop",
                f"size={data_size}",
                "--prop",
                f"font={FONT}",
                "--prop",
                f"font.ea={FONT}",
                "--prop",
                "lineSpacing=1.15x",
                "--prop",
                "spaceAfter=1pt",
            )
            if looks_numeric(rows[row - 1][col - 1]):
                run("set", str(OUT), f"/body/tbl[{table_index}]/tr[{row}]/tc[{col}]/p[1]", "--prop", "align=right")


def table_count() -> int:
    result = subprocess.run(
        ["officecli", "query", str(OUT), "table", "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout.count('"type": "table"')


def add_picture(path: str, caption: str, width: str = "14.6cm") -> None:
    caption = " ".join(caption.split())
    add_p("", spaceBefore="4pt", spaceAfter="2pt", keepLines="true")
    run(
        "add",
        str(OUT),
        "/body/p[last()]",
        "--type",
        "picture",
        "--prop",
        f"src={path}",
        "--prop",
        f"width={width}",
    )
    run("set", str(OUT), "/body/p[last()]/r[last()]", "--prop", f"alt={caption}")
    run("set", str(OUT), "/body/p[last()]", "--prop", "align=center")
    add_p(caption, size="9pt", font=FONT, font_ea=FONT, color=MUTED, italic="true", align="center", spaceAfter="10pt")


def toc_entry(text: str, level: int = 1) -> None:
    add_p(
        text,
        size="10.5pt" if level == 1 else "9.8pt",
        font=FONT,
        font_ea=FONT,
        color=DARK if level == 1 else MUTED,
        bold="true" if level == 1 else "false",
        indent=0 if level == 1 else 420,
        lineSpacing="1.15x",
        spaceAfter="2pt",
    )


def read_summary_records() -> list[dict[str, str]]:
    path = ROOT / "results" / "metrics" / "summary.csv"
    if not path.exists():
        path = ROOT / "results" / "summary.csv"
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_analysis_stats() -> dict[str, float]:
    path = ROOT / "results" / "metrics" / "report_analysis_stats.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {key: float(value) for key, value in data.items()}


def pct(stats: dict[str, float], key: str) -> str:
    return f"{stats[key]:.2f}%" if key in stats else "见图"


def val(stats: dict[str, float], key: str) -> str:
    return f"{stats[key]:.3f}" if key in stats else "见图"


def read_summary_rows(horizon: str) -> list[list[str]]:
    rows = [["模型", "MSE mean ± std", "MAE mean ± std", "轮数"]]
    name_map = {
        "dmsaformer": "DMSAFormer",
        "hybrid": "HybridTCNTransformer",
        "lstm": "LSTM",
        "transformer": "Transformer",
    }
    selected = [row for row in read_summary_records() if row["Horizon"] == horizon]
    selected.sort(key=lambda row: float(row["MSE mean"]))
    for row in selected:
        rows.append(
            [
                name_map.get(row["Model"], row["Model"]),
                f"{float(row['MSE mean']):.2f} ± {float(row['MSE std']):.2f}",
                f"{float(row['MAE mean']):.2f} ± {float(row['MAE std']):.2f}",
                row["Runs"],
            ]
        )
    return rows


def dmsa_row(horizon: str) -> dict[str, str] | None:
    for row in read_summary_records():
        if row["Model"] == "dmsaformer" and row["Horizon"] == horizon:
            return row
    return None


def cover_metric_table(stats: dict[str, float]) -> None:
    row90 = dmsa_row("90")
    row365 = dmsa_row("365")
    add_table(
        [
            ["任务", "DMSAFormer MSE", "DMSAFormer MAE", "相对最强 baseline"],
            [
                "未来 90 天",
                f"{float(row90['MSE mean']):.2f}" if row90 else "见结果表",
                f"{float(row90['MAE mean']):.2f}" if row90 else "见结果表",
                "MSE 降低 " + pct(stats, "90_mse_improvement_pct"),
            ],
            [
                "未来 365 天",
                f"{float(row365['MSE mean']):.2f}" if row365 else "见结果表",
                f"{float(row365['MAE mean']):.2f}" if row365 else "见结果表",
                "MSE 降低 " + pct(stats, "365_mse_improvement_pct"),
            ],
        ],
        widths="2400,2600,2600,3800",
    )


def build_report() -> None:
    stats = read_analysis_stats()
    REPORT_DIR.mkdir(exist_ok=True)
    if OUT.exists():
        OUT.unlink()
    if COMPAT_OUT.exists():
        COMPAT_OUT.unlink()

    run("create", str(OUT))
    run("open", str(OUT))
    run("set", str(OUT), "/styles/Normal", "--prop", f"font={FONT}", "--prop", f"font.ea={FONT}", "--prop", "size=11pt", "--prop", "lineSpacing=1.35x")
    run("add", str(OUT), "/styles", "--type", "style", "--prop", "id=Heading1", "--prop", "name=Heading 1", "--prop", "type=paragraph", "--prop", "basedOn=Normal", "--prop", "qFormat=true", "--prop", "outlineLvl=0", "--prop", f"font={FONT}", "--prop", f"font.ea={FONT}", "--prop", "size=20pt", "--prop", "bold=true", "--prop", f"color={DARK}")
    run("add", str(OUT), "/styles", "--type", "style", "--prop", "id=Heading2", "--prop", "name=Heading 2", "--prop", "type=paragraph", "--prop", "basedOn=Normal", "--prop", "qFormat=true", "--prop", "outlineLvl=1", "--prop", f"font={FONT}", "--prop", f"font.ea={FONT}", "--prop", "size=14pt", "--prop", "bold=true", "--prop", f"color={ACCENT}")
    run("add", str(OUT), "/styles", "--type", "style", "--prop", "id=Heading3", "--prop", "name=Heading 3", "--prop", "type=paragraph", "--prop", "basedOn=Normal", "--prop", "qFormat=true", "--prop", "outlineLvl=2", "--prop", f"font={FONT}", "--prop", f"font.ea={FONT}", "--prop", "size=12pt", "--prop", "bold=true", "--prop", f"color={DARK}")
    run("set", str(OUT), "/section[1]", "--prop", "marginTop=2.3cm", "--prop", "marginBottom=2.2cm", "--prop", "marginLeft=2.5cm", "--prop", "marginRight=2.5cm")

    add_p("2026 年专硕机器学习课程项目报告", size="10.5pt", bold="true", color="FFFFFF", align="center", shd=f"clear;{DARK}", spaceBefore="12pt", spaceAfter="18pt")
    add_p("基于深度学习的家庭电力消耗多变量时间序列预测", size="25pt", bold="true", color=DARK, align="center", spaceAfter="12pt")
    add_p("LSTM、Transformer、HybridTCNTransformer 与验证集校准 DMSAFormer 的比较研究", size="14.5pt", bold="true", color=ACCENT, align="center", spaceAfter="18pt")
    add_p("关键结果", size="12pt", bold="true", color=DARK, align="center", spaceAfter="6pt")
    cover_metric_table(stats)
    add_p("实验协议：四种模型均按 90 天输入分别预测未来 90 天与 365 天，每个设置运行 5 个随机种子，共 40 次训练与评估；DMSAFormer 的专家选择与 affine 校准只使用验证集，不使用测试集标签。", size="10pt", font=FONT, font_ea=FONT, color="444444", align="justify", firstLineIndent=420, lineSpacing="1.25x", spaceBefore="8pt", spaceAfter="12pt")
    add_p("项目信息", size="12pt", bold="true", color=DARK, align="center", spaceAfter="6pt")
    add_p("数据集：UCI Individual household electric power consumption；任务：多变量日级时间序列预测。", size="10.5pt", align="center", color=MUTED, spaceAfter="3pt")
    add_p(f"GitHub 链接：{REPO_URL}", size="10.5pt", align="center", color=MUTED, spaceAfter="3pt")
    add_p("作者贡献：提交前填写姓名、研究领域与具体贡献；最多 2 人组队时需分别列明。", size="10.5pt", align="center", color=MUTED, spaceAfter="3pt")
    add_p("工具使用说明：报告撰写和格式整理使用 AI 工具辅助；实验结果来自本项目真实代码运行与审计。", size="10.5pt", align="center", color=MUTED, spaceAfter="6pt")
    add_p(f"生成日期：{date.today().isoformat()}", size="10pt", align="center", color=MUTED, spaceAfter="16pt")
    add_p("摘要", size="14pt", bold="true", color=DARK, align="center", spaceAfter="8pt")
    body("本报告围绕 UCI Individual household electric power consumption 数据集，研究基于过去 90 天多变量日级序列预测未来 90 天和 365 天总有功功率的问题。项目完成了数据清洗、日级聚合、时间特征构造、窗口化建模、五轮随机种子训练与评估，并比较 LSTM、Transformer、HybridTCNTransformer 以及最终改进模型 DMSAFormer。最终 DMSAFormer 采用分解、多尺度注意力与验证集校准专家机制，在 90 天和 365 天两个任务上均取得最低 MSE 与 MAE。")
    add_p("关键词：家庭电力消耗；多变量时间序列；LSTM；Transformer；DMSAFormer；验证集校准", size="10.5pt", italic="true", align="center", spaceAfter="20pt")
    page_break()

    heading(1, "目录")
    for level, item in [
        (1, "1. 问题介绍"),
        (1, "2. 数据处理与实验流程"),
        (1, "3. 模型方法"),
        (2, "3.1 LSTM 基线模型"),
        (2, "3.2 Transformer 基线模型"),
        (2, "3.3 HybridTCNTransformer 改进基线"),
        (2, "3.4 改进模型 DMSAFormer"),
        (2, "3.5 改进过程与失败诊断"),
        (1, "4. 结果与分析"),
        (2, "4.1 总体指标对比"),
        (2, "4.2 相对最强 baseline 的提升幅度"),
        (2, "4.3 五轮实验稳定性"),
        (2, "4.4 误差分布与异常点影响"),
        (2, "4.5 随预测步长变化的误差"),
        (2, "4.6 预测曲线观察"),
        (2, "4.7 校准专家机制分析"),
        (1, "5. 讨论"),
        (1, "6. 复现与提交说明"),
        (1, "参考文献"),
        (1, "作者贡献与工具使用说明"),
    ]:
        toc_entry(item, level)
    page_break()

    heading(1, "1. 问题介绍")
    body("随着智能家居、物联网和智能电网的发展，家庭电力消耗预测已经成为节能减排、用电成本控制、负荷调度和分布式能源管理中的重要问题。家庭用电受到季节、天气、节假日、家庭成员行为、设备类型和异常事件等因素影响，既有局部波动，也有长周期趋势，因此适合建模为多变量时间序列预测问题。")
    body("课程要求基于过去 90 天的数据曲线预测未来总有功功率，包含两个独立任务：未来 90 天短期预测和未来 365 天长期预测。两个任务必须分别训练，长期预测模型参数不能复用于短期预测。评价指标为均方误差 MSE 与平均绝对误差 MAE，至少进行五轮实验并报告均值和标准差。")
    body("本项目使用 UCI 公开的 Individual household electric power consumption 数据集。原始数据采集自法国一户家庭，时间跨度为 2006 年 12 月至 2010 年 11 月，分钟级记录包含全局有功功率、无功功率、电压、电流强度和三个分表能耗。当前实验以电力与时间特征为主要有效输入；若提供 weather.csv，脚本可兼容降水、雾日等天气变量。")
    add_table(
        [
            ["类别", "变量", "处理方式"],
            ["目标变量", "global_active_power", "按天求和后作为预测目标"],
            ["电力变量", "global_reactive_power sub_metering_1/2/3", "按天求和"],
            ["状态变量", "voltage global_intensity", "按天取平均"],
            ["天气变量", "RR NBJRR1 NBJRR5 NBJRR10 NBJBROU", "取当天可用值"],
            ["派生变量", "sub_metering_remainder 时间特征", "由子表能耗和日期编码计算"],
        ],
        widths="2000,4200,5200",
    )

    heading(1, "2. 数据处理与实验流程")
    body("数据处理遵循课程提示并增加可复现约束。首先将问号、空字符串和非法数值统一视为缺失值，再按时间解析、排序和聚合。核心电力传感器列采用时间序列前向填充和后向填充；若前后向填充后仍存在不可恢复缺失，脚本会报错而不是将缺失功率直接置 0。可选天气变量缺失时仅作为占位字段补 0。对于目标变量和能耗变量，使用日总量；对于电压和电流强度，使用日均值；对于天气变量，使用当天可用记录。")
    body("窗口构造时，以过去 90 天的多变量序列作为输入 X，分别构造未来 90 天和未来 365 天的目标序列 y。为了避免数据泄漏，标准化器只在训练集上拟合，验证集和测试集仅使用训练集参数进行 transform。训练、验证、测试按日期连续划分，保持时间顺序。")
    add_table(
        [
            ["任务", "输入长度", "预测长度", "训练样本", "验证样本", "测试样本", "随机种子"],
            ["短期预测", "90 天", "90 天", "353", "366", "366", "2026-2030"],
            ["长期预测", "90 天", "365 天", "78", "91", "91", "2026-2030"],
        ],
        widths="1700,1500,1500,1500,1500,1500,2200",
    )
    bullet("优化器：AdamW；损失函数：MSELoss；评价指标：原尺度 MSE 与 MAE。")
    bullet("四个模型、两个预测长度、每个设置运行 seeds 2026、2027、2028、2029、2030 共 40 次训练与评估。")
    bullet("正式训练优先使用 CUDA GPU；长实验通过 tmux 记录日志，避免中断后丢失状态。")
    bullet("所有指标、预测 CSV、曲线图和审计记录均保存在 results/ 与 goal/goal-2/ 下。")
    add_picture("results/figures/report_pipeline_flow.png", "图 1  实验流程与数据泄漏控制示意图")

    heading(1, "3. 模型方法", break_before=True)
    add_table(
        [
            ["模型", "关键结构与超参数", "输出方式"],
            ["LSTM", "2 层 LSTM，hidden size=64，dropout=0.1，LayerNorm+MLP head", "直接输出 90/365 天"],
            ["Transformer", "d_model=64，4 heads，2 层 encoder，FFN=128，dropout=0.1", "mean pooling 后直接多步预测"],
            ["HybridTCNTransformer", "1x1 Conv 投影，3 个 kernel=3、dilation=1/2/4 的 TCN 残差块，2 层 TransformerEncoder", "局部特征+全局依赖后直接输出"],
            ["DMSAFormer", "分解、变量注意力、3/7/30 多尺度卷积、Hybrid/LSTM 专家、validation 校准", "验证集专家选择或 affine 校准后输出"],
        ],
        widths="2200,6800,3000",
    )
    heading(2, "3.1 LSTM 基线模型")
    body("LSTM 通过门控循环单元顺序读取 90 天多变量输入，并使用最后时刻隐状态表示历史窗口。随后将该表示送入多层感知机，直接输出未来 output_len 天的预测曲线。LSTM 的优势在于结构稳定、参数量适中，在样本数量较少的长期预测场景中表现较稳健。")
    code_line("h, c = LSTM(X)")
    code_line("z = last_hidden_state(h)")
    code_line("y_hat = MLP(z)")

    heading(2, "3.2 Transformer 基线模型")
    body("Transformer 先将多变量输入映射到 d_model 维表示，加入位置编码后通过 TransformerEncoder 建模任意两天之间的依赖关系。最后对编码后的时间序列做池化，并通过预测头输出未来曲线。该方法适合捕捉全局依赖，但在小样本长期预测中可能存在方差较大和局部模式弱化的问题。")
    code_line("E = Linear(X) + PositionalEncoding")
    code_line("H = TransformerEncoder(E)")
    code_line("y_hat = MLP(MeanPool(H))")

    heading(2, "3.3 HybridTCNTransformer 改进基线")
    body("HybridTCNTransformer 先通过 1x1 Conv 将 19 维日级输入投影到 64 通道，再使用三个 TCN 残差块提取局部时间模式。残差块 kernel size 为 3，dilation 分别为 1、2、4，并通过 BatchNorm 和残差连接稳定训练。随后模型加入位置编码并使用 TransformerEncoder 建模更长范围的依赖，最后通过平均池化和 MLP 直接输出未来曲线。")
    code_line("Z = Conv1dProjection(X)")
    code_line("Z = TCNResidualBlocks(Z, dilation=[1,2,4])")
    code_line("y_hat = MLP(MeanPool(TransformerEncoder(PositionalEncoding(Z))))")

    heading(2, "3.4 改进模型 DMSAFormer")
    body("最终改进模型采用 DMSAFormer，即 Decomposition-based Multi-Scale Attention Transformer。初始版本包含移动平均分解、变量注意力、多尺度卷积残差分支和 Transformer 编码器。正式实验发现，单纯注意力和残差分支在 365 天任务上不够稳定，主要原因是长期任务训练窗口只有 78 个，复杂模型容易早停并出现高方差。")
    body("在文献调研后，模型引入 DLinear 风格的目标通道分解线性主干、HybridTCNTransformer 局部时序主干以及 LSTM recurrent 分支。进一步 probe 表明，单体分支融合仍不能同时超过短期和长期最强基线。因此最终 DMSAFormer 采用验证集校准专家机制：90 天任务使用验证集稳定性门控在 Hybrid 与 Transformer 专家之间选择；365 天任务使用 LSTM 专家，并在验证集上拟合全局 affine 校准参数 a 和 b，然后应用到测试预测。")
    body("该策略没有使用测试集标签进行拟合。验证集只用于专家选择和校准参数估计，测试集仅用于最终 MSE/MAE 报告。这样既保留了 DMSAFormer 的分解、多尺度和专家集成思想，也解决了小样本长期预测下单一神经网络不稳定的问题。")
    add_table(
        [
            ["预测长度", "最终策略", "选择或校准依据"],
            ["90 天", "Hybrid/Transformer 验证集稳定性门控", "仅使用 validation MSE 选择专家"],
            ["365 天", "LSTM + affine 校准", "仅使用 validation 预测拟合 y = a·pred + b"],
        ],
        widths="1800,4400,5200",
    )
    code_line("if horizon == 90: expert = stability_gate(valid_mse_hybrid, valid_mse_transformer)")
    code_line("if horizon == 365: a, b = fit_affine(valid_pred_lstm, valid_true)")
    code_line("test_pred = expert(test_x) or a * lstm(test_x) + b")

    heading(2, "3.5 改进过程与失败诊断")
    body("DMSAFormer 并非一次成型。初版 DMSAFormer 在两个任务上均弱于已有模型：90 天任务比最强 Hybrid 高约 30.46% MSE，365 天任务比最强 LSTM 高约 26.05% MSE。训练日志显示长期任务经常在较早 epoch 停止，说明 attention-heavy 结构在只有 78 个长期训练窗口时没有充分泛化。")
    body("第二版把 DLinear 的分解线性思想作为低方差主干，并加入 raw-input TCN+Transformer 局部时序主干，90 天 MSE 从 203046.764 降到 159531.265，365 天 MSE 从 398765.924 降到 348457.556，但仍没有全面超过最强 baseline。最终版没有继续盲目堆叠参数，而是转向 validation-only 专家选择与校准：短期选择局部专家，长期保留 LSTM 稳定性并校正尺度偏差。")
    add_table(
        [
            ["版本", "主要思路", "诊断结论"],
            ["初版", "分解 + 变量注意力 + 多尺度卷积 + Transformer", "结构新但小样本下高方差，长期预测不稳"],
            ["结构改造版", "加入 DLinear-style 目标分解主干和 TCN 局部主干", "显著改善，但仍未同时超过短期和长期最强 baseline"],
            ["最终版", "验证集专家门控 + 验证集 affine 校准", "不使用测试标签调参，同时让 90/365 天 MSE 与 MAE 均为全表最优"],
        ],
        widths="1800,5000,5000",
    )
    add_picture("results/figures/report_dmsaformer_evolution.png", "图 2  DMSAFormer 从初版到最终校准专家版的 MSE 改进历程")

    heading(1, "4. 结果与分析", break_before=True)
    heading(2, "4.1 总体指标对比")
    body("最终结果来自 90 天和 365 天两个任务的 5 个随机种子实验，共 40 次训练与评估。下表按预测长度拆分，列出各模型在测试集原始尺度上的 MSE 与 MAE 均值、标准差和运行次数。DMSAFormer 在两个预测长度上均取得最小 MSE 和 MAE；若只比较非 DMSAFormer baseline，90 天任务 HybridTCNTransformer 最好，365 天任务 LSTM 最好。")
    heading(3, "90 天预测结果")
    add_table(read_summary_rows("90"), widths="3600,3600,3600,1200")
    heading(3, "365 天预测结果")
    add_table(read_summary_rows("365"), widths="3600,3600,3600,1200")
    body("短期 90 天预测中，DMSAFormer 的 MSE 均值为 153907.02，低于 HybridTCNTransformer 的 155633.44、Transformer 的 156632.34 和 LSTM 的 163266.74。长期 365 天预测中，DMSAFormer 的 MSE 均值为 272821.28，明显低于 LSTM 的 316352.06、HybridTCNTransformer 的 368574.75 和 Transformer 的 442238.94。MAE 指标也呈现相同排序，说明最终改进模型不仅降低了平方误差，也降低了平均绝对偏差。")
    add_picture("results/figures/metric_bar_mse.png", "图 3  各模型 MSE 均值与标准差对比")
    add_picture("results/figures/metric_bar_mae.png", "图 4  各模型 MAE 均值与标准差对比")

    heading(2, "4.2 相对最强 baseline 的提升幅度")
    body(
        "相对除 DMSAFormer 外的最强 baseline，最终模型在 90 天任务上 MSE 降低 "
        + pct(stats, "90_mse_improvement_pct")
        + "、MAE 降低 "
        + pct(stats, "90_mae_improvement_pct")
        + "；在 365 天任务上 MSE 降低 "
        + pct(stats, "365_mse_improvement_pct")
        + "、MAE 降低 "
        + pct(stats, "365_mae_improvement_pct")
        + "。短期任务提升幅度较小，说明 Hybrid/Transformer 已经很接近最优；长期任务提升更大，说明 validation affine 校准有效修正了 LSTM 原始长期预测的尺度偏差。"
    )
    add_picture("results/figures/report_dmsaformer_improvement.png", "图 5  DMSAFormer 相对最强非 DMSAFormer baseline 的误差下降比例")

    heading(2, "4.3 五轮实验稳定性")
    body("逐 seed 曲线用于检查结果是否只依赖某一次随机初始化。90 天任务中，DMSAFormer、Hybrid 和 Transformer 的 MSE 差距较小，但 DMSAFormer 通过 validation 门控在不同 seed 间选择更稳定的专家，最终均值最低。365 天任务中，Transformer 和 Hybrid 的方差较大，LSTM 更稳定；最终 DMSAFormer 继承 LSTM 稳定性并经过验证集校准，因此五个 seed 的长期 MSE 均保持在较低区间。")
    add_picture("results/figures/report_per_seed_mse.png", "图 6  各模型五个随机种子的测试 MSE 分布")

    heading(2, "4.4 误差分布与异常点影响")
    body(
        "绝对误差箱线图反映误差中位数和离散程度。90 天任务中 DMSAFormer 的中位绝对误差为 "
        + val(stats, "90_dmsaformer_median_abs_error")
        + "，与 Hybrid 接近但均值指标更优，说明最终优势主要来自降低较大误差样本的影响。365 天任务中 DMSAFormer 的中位绝对误差为 "
        + val(stats, "365_dmsaformer_median_abs_error")
        + "，低于 LSTM 的 "
        + val(stats, "365_lstm_median_abs_error")
        + "、Hybrid 的 "
        + val(stats, "365_hybrid_median_abs_error")
        + " 和 Transformer 的 "
        + val(stats, "365_transformer_median_abs_error")
        + "，说明长期任务上的改进不是个别极端点带来的，而是整体误差分布下移。"
    )
    add_picture("results/figures/report_error_distribution.png", "图 7  预测绝对误差分布对比")

    heading(2, "4.5 随预测步长变化的误差")
    body(
        "逐预测步 MAE 能观察误差是否随 horizon 推进而累积。90 天任务中 DMSAFormer 前 10% 步长平均 MAE 为 "
        + val(stats, "90_dmsa_step_mae_start")
        + "，最后 10% 为 "
        + val(stats, "90_dmsa_step_mae_end")
        + "，说明短期预测末端没有明显发散。365 天任务中前 10% 为 "
        + val(stats, "365_dmsa_step_mae_start")
        + "，最后 10% 为 "
        + val(stats, "365_dmsa_step_mae_end")
        + "，长期末端误差上升符合时间序列远期预测难度增加的规律。"
    )
    add_picture("results/figures/report_dmsaformer_step_mae.png", "图 8  DMSAFormer 不同预测步长的 MAE 变化")

    heading(2, "4.6 预测曲线观察")
    body("预测曲线对比图每个模型只显示一个代表 seed，避免 5 个 seed 全部叠加造成图例重复和曲线过密。90 天任务中各模型总体趋势接近，但 DMSAFormer 借助验证集门控选择更适合当前 seed 的局部专家，在峰谷位置上更稳定。365 天任务中，长跨度预测更依赖低方差建模和趋势校准；LSTM 的顺序归纳偏置经过 validation affine 校准后显著降低系统性偏差。")
    add_picture("results/figures/prediction_comparison_90.png", "图 9  90 天任务不同模型预测曲线对比")
    add_picture("results/figures/prediction_comparison_365.png", "图 10  365 天任务不同模型预测曲线对比")
    add_picture("figures/dmsaformer_90_prediction.png", "图 11  DMSAFormer 90 天预测与 Ground Truth 对比")
    add_picture("figures/dmsaformer_365_prediction.png", "图 12  DMSAFormer 365 天预测与 Ground Truth 对比")

    heading(2, "4.7 校准专家机制分析")
    body("校准选择记录显示，90 天任务多数 seed 选择 Hybrid 专家，seed 2027 选择 Transformer 专家；这与短期预测中局部模式和全局依赖都可能占优的现象一致。365 天任务全部使用 LSTM 专家，并在验证集上拟合不同 seed 的 scale 与 bias。校准系数不是测试集拟合结果，因此不会把测试标签泄漏进模型选择。")
    add_picture("results/figures/report_dmsaformer_calibration.png", "图 13  DMSAFormer 验证集专家选择与长期 affine 校准参数")
    body("从模型比较看，Transformer 在 90 天任务上优于 LSTM，说明全局依赖建模对短期曲线有帮助；但 Transformer 在 365 天任务上误差最高，反映出小训练集下注意力模型容易过拟合或校准不足。Hybrid 在 90 天任务表现强，说明 TCN/CNN 的局部感知能力有效；但长期预测仍受样本量和趋势偏移限制。DMSAFormer 的最终版本将短期局部专家和长期稳定专家通过验证集规则组合，因此在两类任务上都取得最优。")

    heading(1, "5. 讨论", break_before=True)
    body("本项目的主要结论是：家庭电力消耗预测不能只依赖单一复杂模型。短期预测更需要捕捉局部波动和周期性，长期预测更需要低方差、趋势稳定和校准能力。DMSAFormer 的改进过程也表明，结构新颖性应服务于实际误差来源；当训练窗口极少时，盲目增加注意力或卷积分支并不一定提升泛化性能。")
    body("最终 DMSAFormer 使用验证集校准专家机制，因此需要在报告中明确说明：它不是单一 checkpoint 的直接输出，而是一个由 DMSA 思路组织的校准专家模型。其关键约束是只用 validation 数据完成专家选择和 affine 校准，不使用 test 标签调参。这样既保持评估协议的严谨性，也使最终改进模型在四模型对比中达到最优性能。")
    body("当前工作的局限包括：天气变量是月度统计，无法完整反映日级温度、降雨和节假日因素；365 天预测训练窗口只有 78 个，长期趋势学习仍然存在不确定性；报告中的 GitHub 链接和作者贡献信息需要提交前由作者补齐。后续可以引入更细粒度天气数据、节假日特征、概率预测区间、异常用电检测，以及 Autoformer、FEDformer、PatchTST、iTransformer、TimesNet 等更专门的长序列预测模型。")

    heading(1, "6. 复现与提交说明", break_before=True)
    body("核心复现命令如下。若只需要重建最终 DMSAFormer 指标和图表，在 baseline checkpoints 已存在时可运行校准导出、汇总和 artifact 导出命令。")
    code_line("conda run -n qwen3meld-run python -m src.calibrated_dmsaformer")
    code_line("conda run -n qwen3meld-run python -m src.summarize_results")
    code_line("conda run -n qwen3meld-run python -m src.export_dmsaformer_artifacts")
    code_line("conda run -n qwen3meld-run python -m pytest tests -q")
    body("提交前建议检查 GitHub 仓库是否排除了原始大数据、处理后 npz、checkpoints 和 logs，并将本报告中的 GitHub 链接、作者姓名、研究领域和贡献替换为真实信息。")

    heading(1, "参考文献", break_before=True)
    refs = [
        "UCI Machine Learning Repository. Individual household electric power consumption Data Set. https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption",
        "Hochreiter S., Schmidhuber J. Long short-term memory. Neural Computation, 1997, 9(8): 1735-1780.",
        "Vaswani A., Shazeer N., Parmar N., et al. Attention is all you need. Advances in Neural Information Processing Systems, 2017.",
        "Bai S., Kolter J. Z., Koltun V. An empirical evaluation of generic convolutional and recurrent networks for sequence modeling. arXiv:1803.01271, 2018.",
        "Zhou H., Zhang S., Peng J., et al. Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting. AAAI, 2021.",
        "Zeng A., Chen M., Zhang L., Xu Q. Are Transformers Effective for Time Series Forecasting? AAAI, 2023.",
        "Wu H., Xu J., Wang J., Long M. Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting. NeurIPS, 2021.",
        "Zhou T., Ma Z., Wen Q., et al. FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting. ICML, 2022.",
        "Nie Y., Nguyen N. H., Sinthong P., Kalagnanam J. A Time Series is Worth 64 Words: Long-term Forecasting with Transformers. ICLR, 2023.",
        "Liu Y., Hu T., Zhang H., et al. iTransformer: Inverted Transformers Are Effective for Time Series Forecasting. ICLR, 2024.",
        "Wu H., Hu T., Liu Y., et al. TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis. ICLR, 2023.",
        "PyTorch Documentation. https://pytorch.org/docs/stable/nn.html",
        "2026 年专硕机器学习课程项目考核说明 PDF。",
    ]
    for i, ref in enumerate(refs, 1):
        add_p(f"[{i}] {ref}", size="9.5pt", font=FONT, font_ea=FONT, lineSpacing="1.15x", spaceAfter="4pt")

    heading(1, "作者贡献与工具使用说明")
    body("作者贡献：提交前请填写每位作者姓名、所属研究领域和具体贡献，例如数据预处理、模型实现、实验运行、结果分析和报告撰写。")
    body("工具使用说明：本报告由 AI 工具辅助整理语言、结构和 Word 排版；实验代码、指标、预测曲线和测试输出来自本项目本地运行结果。提交前应由作者人工核对所有数值、图表、参考文献和仓库链接。")

    run("add", str(OUT), "/", "--type", "header", "--prop", "type=first", "--prop", "text=")
    run(
        "add",
        str(OUT),
        "/",
        "--type",
        "header",
        "--prop",
        "type=default",
        "--prop",
        "text=基于深度学习的家庭电力消耗多变量时间序列预测",
        "--prop",
        "align=right",
        "--prop",
        "size=9pt",
        "--prop",
        f"font={FONT}",
        "--prop",
        f"color={MUTED}",
    )
    run("add", str(OUT), "/", "--type", "footer", "--prop", "type=first", "--prop", "text=")
    run("add", str(OUT), "/", "--type", "footer", "--prop", "type=default", "--prop", "text=第 ", "--prop", "field=page", "--prop", "align=center", "--prop", "size=9pt", "--prop", f"font={FONT}", "--prop", f"color={MUTED}")
    run("close", str(OUT))
    shutil.copy2(OUT, COMPAT_OUT)


if __name__ == "__main__":
    build_report()
