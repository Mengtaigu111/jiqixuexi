# PDF 转 Word 重做计划

## 需求规格
- 目标：重新生成更可靠的 Word 文档，修复现有 Word 中文字识别、字体和版式问题。
- 输入：优先分析 `report.pdf` / `ML_household_power_report.pdf`、现有 `ML_household_power_report.docx`，并利用同目录的 `report.html` / `report_draft.md` 作为高质量源材料。
- 输出：生成新的 `.docx`，保留原文件，避免覆盖用户现有版本，除非最终确认需要替换。
- 质量标准：文字应来自可编辑源文本而非 OCR 猜测；中文字体应设置为可用 CJK 字体；标题层级、正文、图片/图表、分页应尽量贴近 PDF。

## 执行计划
- [x] 1. 回顾已有文件和历史教训，确认源文件关系与现有 Word 问题。
- [x] 2. 检查本机可用转换工具、字体和文档结构，选择最稳的生成方案。
- [x] 3. 重建 Word 文档，优先从 HTML/Markdown 源生成，必要时补充图片与样式控制。
- [x] 4. 验证生成结果：检查可编辑文本、中文字体、章节数量、图片数量和基本版式。
- [x] 5. 归档结果复盘，记录生成文件路径、验证命令和后续建议。

## 改动说明
- 初始化任务计划，先避免直接覆盖原始 Word。
- 确认 `report.pdf` 由 Chromium 从 HTML 生成，`report.html` 是比 PDF OCR 更可靠的 Word 源。
- 本机没有 `pandoc` / `libreoffice` / `python-docx`，但有 Python、Pillow、PDF 工具和 Noto CJK 字体；采用标准库生成 OOXML，可控设置中文字体、标题、表格和图片。
- 新增 `tools/rebuild_word_report.py`，从 `report.html` 重建可编辑 DOCX。
- 生成 `ML_household_power_report_rebuilt.docx`，保留原 `ML_household_power_report.docx` 不覆盖。
- 验证新文档 ZIP 包无错误，核心 XML 可解析，包含 2 个表格、8 张图片、150 个段落、11370 个可编辑字符，并且 `styles.xml` 含 `Noto Sans CJK SC`。

## 结果复盘
- 输出文件：`ML_household_power_report_rebuilt.docx`。
- 生成方式：不从 PDF 做 OCR/版面猜测，而是使用与 PDF 对应的 `report.html` 作为语义源，显式重建 Word 样式、正文、目录、表格、图片和图注。
- 质量改进：可编辑文本来自 HTML 源；中文字体显式设置为 `Noto Sans CJK SC`；原 Word 未覆盖，可按需人工对比后替换。
- 验证命令：`unzip -t ML_household_power_report_rebuilt.docx`、Python XML 解析与计数检查。

## 错误记录
| 错误 | 处理 |
|---|---|
| 暂无 | 暂无 |

---

# Word 参考文献格式修订

## 需求规格
- 目标：修正 `ML_household_power_report_rebuilt.docx` 中参考文献格式怪异的问题。
- 约束：不能重新从 HTML 生成整篇 Word，避免覆盖用户已插入的终端截图；只在现有 DOCX 上做参考文献段落样式修复。
- 预期格式：参考文献每条独立成段，左对齐，9pt 左右，行距紧凑，编号悬挂缩进，英文和链接可正常换行。

## 执行计划
- [x] 1. 检查现有 DOCX 结构、图片数量和参考文献段落样式。
- [x] 2. 对现有 DOCX 做安全备份。
- [x] 3. 直接修改 OOXML 中参考文献段落样式，保留图片和正文。
- [x] 4. 验证 DOCX 可解压、XML 可解析、图片数量不变、参考文献段落格式已更新。

## 改动说明
- 已确认用户当前文件含 11 张图片，说明插入的终端截图已经在 DOCX 内；本轮不重新生成整篇文档。
- 参考文献当前使用 `Reference` 样式（styleId `14`），但英文也被 run-level 字体固定成 `Noto Sans CJK SC`，会导致英文参考文献观感不自然。
- 已备份原文件为 `ML_household_power_report_rebuilt.before_ref_format.docx`。
- 已原地修正 `ML_household_power_report_rebuilt.docx`：参考文献样式改为英文 `Times New Roman`、中文 `Noto Sans CJK SC`、9pt、左对齐、单倍行距、0.5 英寸悬挂缩进。
- 没有重新生成整篇文档，因此保留用户插入的截图和其他手工编辑。

## 结果复盘
- `unzip -t ML_household_power_report_rebuilt.docx` 通过，无压缩结构错误。
- `word/document.xml`、`word/styles.xml` 等核心 XML 可解析。
- 文档内图片数量仍为 11，未丢失用户新增截图。
- 参考文献数量为 10 条，编号 `[1]` 到 `[10]` 均存在。
- 参考文献样式 `Reference` 已更新为更规范的文献段落格式。
