#!/usr/bin/env python3
"""Rebuild the report DOCX from the HTML source with controlled Word styles."""

from __future__ import annotations

import argparse
import datetime as dt
import os
import posixpath
import re
import shutil
import zipfile
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable
from urllib.parse import unquote, urlparse
from xml.sax.saxutils import escape

from PIL import Image


EMU_PER_INCH = 914400
DOC_FONT = "Noto Sans CJK SC"
CODE_FONT = "Consolas"
TITLE = "基于深度学习的家庭电力消耗多变量时间序列预测"
FOOTER = "2026 年专硕机器学习课程项目报告"


@dataclass
class Node:
    tag: str
    attrs: dict[str, str] = field(default_factory=dict)
    children: list["Node | str"] = field(default_factory=list)

    def text(self) -> str:
        parts: list[str] = []

        def walk(item: "Node | str") -> None:
            if isinstance(item, str):
                parts.append(item)
                return
            if item.tag == "br":
                parts.append("\n")
            for child in item.children:
                walk(child)

        walk(self)
        return normalize_text("".join(parts))

    def elements(self) -> list["Node"]:
        return [child for child in self.children if isinstance(child, Node)]


class TreeParser(HTMLParser):
    VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root = Node("document")
        self.stack = [self.root]

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        node = Node(tag.lower(), {key: value or "" for key, value in attrs})
        self.stack[-1].children.append(node)
        if node.tag not in self.VOID_TAGS:
            self.stack.append(node)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        for index in range(len(self.stack) - 1, 0, -1):
            if self.stack[index].tag == tag:
                del self.stack[index:]
                return

    def handle_data(self, data: str) -> None:
        if data:
            self.stack[-1].children.append(data)


def normalize_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()


def find_first(node: Node, tag: str | None = None, class_name: str | None = None) -> Node | None:
    if tag is None or node.tag == tag:
        if class_name is None or class_name in node.attrs.get("class", "").split():
            return node
    for child in node.elements():
        found = find_first(child, tag, class_name)
        if found is not None:
            return found
    return None


def find_all(node: Node, tag: str | None = None) -> list[Node]:
    found: list[Node] = []
    if tag is None or node.tag == tag:
        found.append(node)
    for child in node.elements():
        found.extend(find_all(child, tag))
    return found


def attr_class_contains(node: Node, value: str) -> bool:
    return value in node.attrs.get("class", "").split()


def xml_text(text: str) -> str:
    return escape(text, {'"': "&quot;"})


def run_xml(
    text: str,
    *,
    bold: bool = False,
    italic: bool = False,
    color: str | None = None,
    size: int | None = None,
    font: str | None = None,
) -> str:
    if text == "":
        return ""
    rpr: list[str] = []
    selected_font = font or DOC_FONT
    rpr.append(
        f'<w:rFonts w:ascii="{xml_text(selected_font)}" w:hAnsi="{xml_text(selected_font)}" '
        f'w:eastAsia="{xml_text(selected_font)}" w:cs="{xml_text(selected_font)}"/>'
    )
    if bold:
        rpr.append("<w:b/><w:bCs/>")
    if italic:
        rpr.append("<w:i/><w:iCs/>")
    if color:
        rpr.append(f'<w:color w:val="{color}"/>')
    if size:
        rpr.append(f'<w:sz w:val="{size}"/><w:szCs w:val="{size}"/>')
    rpr_xml = f"<w:rPr>{''.join(rpr)}</w:rPr>"
    return f'<w:r>{rpr_xml}<w:t xml:space="preserve">{xml_text(text)}</w:t></w:r>'


def paragraph_xml(
    text: str | None = None,
    *,
    style: str = "Normal",
    align: str | None = None,
    page_break_before: bool = False,
    keep_next: bool = False,
    spacing_before: int | None = None,
    spacing_after: int | None = None,
    line: int | None = None,
    indent_left: int | None = None,
    indent_hanging: int | None = None,
    runs: list[str] | None = None,
) -> str:
    props = [f'<w:pStyle w:val="{style}"/>']
    if align:
        props.append(f'<w:jc w:val="{align}"/>')
    if page_break_before:
        props.append("<w:pageBreakBefore/>")
    if keep_next:
        props.append("<w:keepNext/>")
    if spacing_before is not None or spacing_after is not None or line is not None:
        attrs = []
        if spacing_before is not None:
            attrs.append(f'w:before="{spacing_before}"')
        if spacing_after is not None:
            attrs.append(f'w:after="{spacing_after}"')
        if line is not None:
            attrs.append(f'w:line="{line}" w:lineRule="auto"')
        props.append(f"<w:spacing {' '.join(attrs)}/>")
    if indent_left is not None or indent_hanging is not None:
        attrs = []
        if indent_left is not None:
            attrs.append(f'w:left="{indent_left}"')
        if indent_hanging is not None:
            attrs.append(f'w:hanging="{indent_hanging}"')
        props.append(f"<w:ind {' '.join(attrs)}/>")
    body = "".join(runs) if runs is not None else run_xml(text or "")
    return f"<w:p><w:pPr>{''.join(props)}</w:pPr>{body}</w:p>"


def page_break_xml() -> str:
    return '<w:p><w:r><w:br w:type="page"/></w:r></w:p>'


def rels_xml(rels: list[tuple[str, str, str]]) -> str:
    rows = [
        f'<Relationship Id="{rid}" Type="{xml_text(rel_type)}" Target="{xml_text(target)}"/>'
        for rid, rel_type, target in rels
    ]
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        + "".join(rows)
        + "</Relationships>"
    )


class DocxBuilder:
    REL_BASE = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"

    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path
        self.body: list[str] = []
        self.doc_rels: list[tuple[str, str, str]] = []
        self.package_files: dict[str, bytes | str] = {}
        self.image_count = 0
        self.docpr_id = 1
        self.next_rel = 1
        self.image_content_types: dict[str, str] = {}
        self.add_doc_rel(f"{self.REL_BASE}/styles", "styles.xml")
        self.add_doc_rel(f"{self.REL_BASE}/settings", "settings.xml")
        self.add_doc_rel(f"{self.REL_BASE}/fontTable", "fontTable.xml")
        self.header_rid = self.add_doc_rel(f"{self.REL_BASE}/header", "header1.xml")
        self.footer_rid = self.add_doc_rel(f"{self.REL_BASE}/footer", "footer1.xml")

    def add_doc_rel(self, rel_type: str, target: str) -> str:
        rid = f"rId{self.next_rel}"
        self.next_rel += 1
        self.doc_rels.append((rid, rel_type, target))
        return rid

    def add_paragraph(self, *args, **kwargs) -> None:
        self.body.append(paragraph_xml(*args, **kwargs))

    def add_page_break(self) -> None:
        self.body.append(page_break_xml())

    def add_table(self, caption: str, rows: list[list[str]]) -> None:
        if caption:
            self.add_paragraph(caption, style="Caption", keep_next=True)
        if not rows:
            return
        col_count = max(len(row) for row in rows)
        table_width = 9360
        col_width = table_width // col_count
        grid = "".join(f'<w:gridCol w:w="{col_width}"/>' for _ in range(col_count))
        row_xml: list[str] = []
        for row_index, row in enumerate(rows):
            cells: list[str] = []
            for value in row:
                shade = '<w:shd w:fill="D9EAF7"/>' if row_index == 0 else ""
                cell_props = (
                    f'<w:tcPr><w:tcW w:w="{col_width}" w:type="dxa"/>{shade}'
                    '<w:vAlign w:val="center"/></w:tcPr>'
                )
                runs = [run_xml(value, bold=row_index == 0, size=19)]
                para = paragraph_xml(style="TableText", align="center", runs=runs)
                cells.append(f"<w:tc>{cell_props}{para}</w:tc>")
            row_xml.append(f"<w:tr>{''.join(cells)}</w:tr>")
        tbl_props = (
            '<w:tblPr><w:tblStyle w:val="TableGrid"/><w:tblW w:w="9360" w:type="dxa"/>'
            '<w:tblLook w:firstRow="1" w:lastRow="0" w:firstColumn="0" w:lastColumn="0" '
            'w:noHBand="0" w:noVBand="1"/>'
            '<w:tblBorders><w:top w:val="single" w:sz="10" w:color="17365D"/>'
            '<w:left w:val="single" w:sz="4" w:color="B7C9D8"/>'
            '<w:bottom w:val="single" w:sz="10" w:color="17365D"/>'
            '<w:right w:val="single" w:sz="4" w:color="B7C9D8"/>'
            '<w:insideH w:val="single" w:sz="4" w:color="B7C9D8"/>'
            '<w:insideV w:val="single" w:sz="4" w:color="B7C9D8"/></w:tblBorders></w:tblPr>'
        )
        self.body.append(f"<w:tbl>{tbl_props}<w:tblGrid>{grid}</w:tblGrid>{''.join(row_xml)}</w:tbl>")
        self.add_paragraph("", spacing_after=160)

    def add_image(self, image_path: Path, caption: str | None = None, *, max_width_in: float = 6.25, max_height_in: float = 4.55) -> None:
        if not image_path.exists():
            self.add_paragraph(f"[缺失图片] {image_path}", style="Caption")
            return
        self.image_count += 1
        suffix = image_path.suffix.lower().lstrip(".") or "png"
        media_name = f"word/media/image{self.image_count}.{suffix}"
        target = f"media/image{self.image_count}.{suffix}"
        self.package_files[media_name] = image_path.read_bytes()
        content_type = "image/jpeg" if suffix in {"jpg", "jpeg"} else f"image/{suffix}"
        self.image_content_types[suffix] = content_type
        rid = self.add_doc_rel(f"{self.REL_BASE}/image", target)
        with Image.open(image_path) as img:
            px_width, px_height = img.size
        display_w = max_width_in
        display_h = display_w * px_height / px_width
        if display_h > max_height_in:
            display_h = max_height_in
            display_w = display_h * px_width / px_height
        cx = int(display_w * EMU_PER_INCH)
        cy = int(display_h * EMU_PER_INCH)
        name = xml_text(image_path.name)
        docpr_id = self.docpr_id
        self.docpr_id += 1
        drawing = f"""
<w:drawing>
  <wp:inline distT="0" distB="0" distL="0" distR="0">
    <wp:extent cx="{cx}" cy="{cy}"/>
    <wp:effectExtent l="0" t="0" r="0" b="0"/>
    <wp:docPr id="{docpr_id}" name="{name}"/>
    <wp:cNvGraphicFramePr><a:graphicFrameLocks noChangeAspect="1"/></wp:cNvGraphicFramePr>
    <a:graphic>
      <a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">
        <pic:pic>
          <pic:nvPicPr><pic:cNvPr id="{docpr_id}" name="{name}"/><pic:cNvPicPr/></pic:nvPicPr>
          <pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>
          <pic:spPr>
            <a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>
            <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
          </pic:spPr>
        </pic:pic>
      </a:graphicData>
    </a:graphic>
  </wp:inline>
</w:drawing>
""".strip()
        self.body.append(paragraph_xml(style="Image", align="center", runs=[f"<w:r>{drawing}</w:r>"]))
        if caption:
            self.add_paragraph(caption, style="Caption", align="center")

    def document_xml(self) -> str:
        sect_pr = f"""
<w:sectPr>
  <w:headerReference w:type="default" r:id="{self.header_rid}"/>
  <w:footerReference w:type="default" r:id="{self.footer_rid}"/>
  <w:pgSz w:w="11906" w:h="16838"/>
  <w:pgMar w:top="964" w:right="1021" w:bottom="1021" w:left="1021" w:header="454" w:footer="454" w:gutter="0"/>
  <w:cols w:space="708"/>
  <w:docGrid w:linePitch="312"/>
</w:sectPr>
""".strip()
        return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
 xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
 xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
 xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">
  <w:body>{''.join(self.body)}{sect_pr}</w:body>
</w:document>
"""

    def write(self) -> None:
        package_rels = [
            ("rId1", f"{self.REL_BASE}/officeDocument", "word/document.xml"),
            ("rId2", "http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties", "docProps/core.xml"),
            ("rId3", f"{self.REL_BASE}/extended-properties", "docProps/app.xml"),
        ]
        files: dict[str, bytes | str] = {
            "[Content_Types].xml": self.content_types_xml(),
            "_rels/.rels": rels_xml(package_rels),
            "docProps/core.xml": core_props_xml(),
            "docProps/app.xml": app_props_xml(),
            "word/document.xml": self.document_xml(),
            "word/_rels/document.xml.rels": rels_xml(self.doc_rels),
            "word/styles.xml": styles_xml(),
            "word/settings.xml": settings_xml(),
            "word/fontTable.xml": font_table_xml(),
            "word/header1.xml": header_xml(),
            "word/footer1.xml": footer_xml(),
            **self.package_files,
        }
        with zipfile.ZipFile(self.output_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for name, data in files.items():
                zf.writestr(name, data)

    def content_types_xml(self) -> str:
        image_defaults = "".join(
            f'<Default Extension="{xml_text(ext)}" ContentType="{xml_text(content_type)}"/>'
            for ext, content_type in sorted(self.image_content_types.items())
        )
        return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  {image_defaults}
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
  <Override PartName="/word/settings.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.settings+xml"/>
  <Override PartName="/word/fontTable.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.fontTable+xml"/>
  <Override PartName="/word/header1.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.header+xml"/>
  <Override PartName="/word/footer1.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.footer+xml"/>
</Types>
"""


def style_xml(style_id: str, name: str, based_on: str | None, ppr: str, rpr: str) -> str:
    based = f'<w:basedOn w:val="{based_on}"/>' if based_on else ""
    return f"""
<w:style w:type="paragraph" w:styleId="{style_id}">
  <w:name w:val="{xml_text(name)}"/>
  {based}
  <w:qFormat/>
  <w:pPr>{ppr}</w:pPr>
  <w:rPr>{rpr}</w:rPr>
</w:style>
""".strip()


def default_rpr(size: int = 21, color: str = "2F3437", font: str = DOC_FONT) -> str:
    return (
        f'<w:rFonts w:ascii="{font}" w:hAnsi="{font}" w:eastAsia="{font}" w:cs="{font}"/>'
        f'<w:color w:val="{color}"/><w:sz w:val="{size}"/><w:szCs w:val="{size}"/>'
    )


def styles_xml() -> str:
    styles = [
        """
<w:style w:type="paragraph" w:default="1" w:styleId="Normal">
  <w:name w:val="Normal"/>
  <w:qFormat/>
  <w:pPr><w:spacing w:after="120" w:line="312" w:lineRule="auto"/><w:jc w:val="both"/></w:pPr>
  <w:rPr>""" + default_rpr() + """</w:rPr>
</w:style>
""".strip(),
        style_xml("CoverKicker", "Cover Kicker", "Normal", '<w:spacing w:after="260"/>', default_rpr(22, "1F4E79") + "<w:b/><w:bCs/>"),
        style_xml("CoverTitle", "Cover Title", "Normal", '<w:spacing w:after="260"/><w:keepNext/>', default_rpr(52, "17365D") + "<w:b/><w:bCs/>"),
        style_xml("CoverSubtitle", "Cover Subtitle", "Normal", '<w:spacing w:after="520"/>', default_rpr(27, "4F626F")),
        style_xml("CoverMeta", "Cover Meta", "Normal", '<w:spacing w:after="90" w:line="360" w:lineRule="auto"/>', default_rpr(21, "3F4E58")),
        style_xml("TOCHeading", "TOC Heading", "Normal", '<w:spacing w:after="360"/><w:keepNext/>', default_rpr(44, "17365D") + "<w:b/><w:bCs/>"),
        style_xml("TOCEntry", "TOC Entry", "Normal", '<w:spacing w:after="120"/><w:ind w:left="240"/>', default_rpr(22, "2F3437")),
        style_xml("Heading1", "Heading 1", "Normal", '<w:spacing w:before="240" w:after="160"/><w:keepNext/>', default_rpr(33, "17365D") + "<w:b/><w:bCs/>"),
        style_xml("Caption", "Caption", "Normal", '<w:spacing w:before="80" w:after="160"/><w:jc w:val="center"/>', default_rpr(18, "4F626F")),
        style_xml("Image", "Image", "Normal", '<w:spacing w:before="160" w:after="80"/><w:jc w:val="center"/>', default_rpr(21, "2F3437")),
        style_xml("Reference", "Reference", "Normal", '<w:spacing w:after="80" w:line="288" w:lineRule="auto"/><w:ind w:left="397" w:hanging="397"/>', default_rpr(18, "2F3437")),
        style_xml("TableText", "Table Text", "Normal", '<w:spacing w:after="0" w:line="260" w:lineRule="auto"/><w:jc w:val="center"/>', default_rpr(19, "2F3437")),
        """
<w:style w:type="table" w:styleId="TableGrid">
  <w:name w:val="Table Grid"/>
  <w:tblPr><w:tblBorders><w:top w:val="single" w:sz="4" w:color="B7C9D8"/><w:left w:val="single" w:sz="4" w:color="B7C9D8"/><w:bottom w:val="single" w:sz="4" w:color="B7C9D8"/><w:right w:val="single" w:sz="4" w:color="B7C9D8"/><w:insideH w:val="single" w:sz="4" w:color="B7C9D8"/><w:insideV w:val="single" w:sz="4" w:color="B7C9D8"/></w:tblBorders></w:tblPr>
</w:style>
""".strip(),
    ]
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:docDefaults>
    <w:rPrDefault><w:rPr>{default_rpr()}</w:rPr></w:rPrDefault>
    <w:pPrDefault><w:pPr><w:spacing w:after="120" w:line="312" w:lineRule="auto"/></w:pPr></w:pPrDefault>
  </w:docDefaults>
  {''.join(styles)}
</w:styles>
"""


def settings_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:settings xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
 xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <w:zoom w:percent="100"/>
  <w:defaultTabStop w:val="420"/>
  <w:characterSpacingControl w:val="doNotCompress"/>
  <w:compat><w:compatSetting w:name="compatibilityMode" w:uri="http://schemas.microsoft.com/office/word" w:val="15"/></w:compat>
</w:settings>
"""


def font_table_xml() -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:fonts xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:font w:name="{DOC_FONT}"><w:panose1 w:val="02010600030101010101"/><w:charset w:val="86"/><w:family w:val="swiss"/><w:pitch w:val="variable"/></w:font>
  <w:font w:name="{CODE_FONT}"><w:family w:val="modern"/><w:pitch w:val="fixed"/></w:font>
</w:fonts>
"""


def header_xml() -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:hdr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  {paragraph_xml(TITLE, style="Caption", align="center")}
</w:hdr>
"""


def footer_xml() -> str:
    runs = [
        run_xml(f"{FOOTER}  "),
        '<w:r><w:fldChar w:fldCharType="begin"/></w:r>',
        '<w:r><w:instrText xml:space="preserve"> PAGE </w:instrText></w:r>',
        '<w:r><w:fldChar w:fldCharType="end"/></w:r>',
    ]
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:ftr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  {paragraph_xml(style="Caption", align="center", runs=runs)}
</w:ftr>
"""


def core_props_xml() -> str:
    now = dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
 xmlns:dc="http://purl.org/dc/elements/1.1/"
 xmlns:dcterms="http://purl.org/dc/terms/"
 xmlns:dcmitype="http://purl.org/dc/dcmitype/"
 xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>{xml_text(TITLE)}</dc:title>
  <dc:creator>Codex</dc:creator>
  <cp:lastModifiedBy>Codex</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>
</cp:coreProperties>
"""


def app_props_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
 xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Codex OOXML Builder</Application>
  <DocSecurity>0</DocSecurity>
  <ScaleCrop>false</ScaleCrop>
  <Company></Company>
  <LinksUpToDate>false</LinksUpToDate>
  <SharedDoc>false</SharedDoc>
  <HyperlinksChanged>false</HyperlinksChanged>
  <AppVersion>16.0000</AppVersion>
</Properties>
"""


def html_path_to_file(src: str, base_dir: Path) -> Path:
    parsed = urlparse(src)
    if parsed.scheme == "file":
        return Path(unquote(parsed.path))
    if parsed.scheme:
        return base_dir / Path(posixpath.basename(parsed.path))
    return (base_dir / unquote(src)).resolve()


def table_rows(table: Node) -> tuple[str, list[list[str]]]:
    caption_node = find_first(table, "caption")
    caption = caption_node.text() if caption_node else ""
    rows: list[list[str]] = []
    for tr in find_all(table, "tr"):
        cells = [cell.text() for cell in tr.elements() if cell.tag in {"th", "td"}]
        if cells:
            rows.append(cells)
    return caption, rows


def add_figure(builder: DocxBuilder, figure: Node, html_dir: Path) -> None:
    image = find_first(figure, "img")
    caption_node = find_first(figure, "figcaption")
    caption = caption_node.text() if caption_node else None
    if image is None:
        return
    image_path = html_path_to_file(image.attrs.get("src", ""), html_dir)
    max_height = 3.65 if image_path.name.startswith("metric_bar") else 4.45
    builder.add_image(image_path, caption, max_height_in=max_height)


def add_node(builder: DocxBuilder, node: Node, html_dir: Path, *, page_break_before: bool = False) -> None:
    if node.tag == "h1":
        builder.add_paragraph(node.text(), style="Heading1", page_break_before=page_break_before, keep_next=True)
    elif node.tag == "p":
        style = "Reference" if attr_class_contains(node, "reference-line") else "Normal"
        text = node.text()
        if text:
            builder.add_paragraph(text, style=style)
    elif node.tag == "table":
        caption, rows = table_rows(node)
        builder.add_table(caption, rows)
    elif node.tag == "figure":
        add_figure(builder, node, html_dir)
    elif node.tag == "div":
        for child in node.elements():
            add_node(builder, child, html_dir)


def build_docx(html_path: Path, output_path: Path) -> None:
    parser = TreeParser()
    parser.feed(html_path.read_text(encoding="utf-8"))
    html_dir = html_path.parent
    builder = DocxBuilder(output_path)

    cover = find_first(parser.root, "section", "cover")
    if cover:
        kicker = find_first(cover, "div", "cover-kicker")
        title = find_first(cover, "h1")
        subtitle = find_first(cover, "div", "cover-subtitle")
        meta = find_first(cover, "div", "cover-meta")
        builder.add_paragraph(kicker.text() if kicker else FOOTER, style="CoverKicker", align="center", spacing_before=1200)
        builder.add_paragraph(title.text() if title else TITLE, style="CoverTitle", align="center")
        builder.add_paragraph(subtitle.text() if subtitle else "", style="CoverSubtitle", align="center")
        if meta:
            for line in meta.elements():
                text = line.text()
                if text:
                    builder.add_paragraph(text, style="CoverMeta", indent_left=900)
        builder.add_page_break()

    toc = find_first(parser.root, "section", "toc-page")
    if toc:
        heading = find_first(toc, "h1")
        builder.add_paragraph(heading.text() if heading else "目录", style="TOCHeading")
        for li in find_all(toc, "li"):
            text = li.text()
            if text:
                builder.add_paragraph(text, style="TOCEntry")
        builder.add_page_break()

    main = find_first(parser.root, "main")
    if main:
        for section in main.elements():
            if section.tag != "section":
                continue
            needs_break = attr_class_contains(section, "page-break-before")
            heading_done = False
            for child in section.elements():
                child_break = needs_break and child.tag == "h1" and not heading_done
                add_node(builder, child, html_dir, page_break_before=child_break)
                if child.tag == "h1":
                    heading_done = True

    builder.write()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--html", type=Path, default=Path("report.html"))
    parser.add_argument("--output", type=Path, default=Path("ML_household_power_report_rebuilt.docx"))
    parser.add_argument("--also-copy-to", type=Path)
    args = parser.parse_args()

    build_docx(args.html.resolve(), args.output.resolve())
    if args.also_copy_to:
        shutil.copy2(args.output, args.also_copy_to)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
