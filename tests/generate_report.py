#!/usr/bin/env python3
"""
PathPatch Navigator - Report Generator
Reads JSON result files from tcga_test_output/ and generates Word (.docx) reports.

Usage:
    python generate_report.py                          # process all JSONs in tcga_test_output/
    python generate_report.py path/to/result.json      # process a specific JSON
    python generate_report.py --output my_reports/     # custom output directory
"""

import json
import sys
import os
import argparse
import subprocess
from pathlib import Path
from datetime import datetime


REPORT_SCRIPT_TEMPLATE = '''
const {{
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  AlignmentType, HeadingLevel, BorderStyle, WidthType, ShadingType,
  LevelFormat, PageNumber, Header, Footer
}} = require('docx');
const fs = require('fs');

const DATA = {data_json};

const border = {{ style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" }};
const borders = {{ top: border, bottom: border, left: border, right: border }};

function heading1(text) {{
  return new Paragraph({{
    heading: HeadingLevel.HEADING_1,
    children: [new TextRun({{ text, bold: true, size: 28, font: "Arial", color: "2E4A7A" }})],
    spacing: {{ before: 360, after: 120 }},
    border: {{ bottom: {{ style: BorderStyle.SINGLE, size: 6, color: "2E4A7A", space: 1 }} }}
  }});
}}

function heading2(text) {{
  return new Paragraph({{
    heading: HeadingLevel.HEADING_2,
    children: [new TextRun({{ text, bold: true, size: 24, font: "Arial", color: "1A5276" }})],
    spacing: {{ before: 240, after: 80 }}
  }});
}}

function para(text, opts = {{}}) {{
  return new Paragraph({{
    children: [new TextRun({{ text, font: "Arial", size: 22, ...opts }})],
    spacing: {{ after: 100 }}
  }});
}}

function infoTable(rows) {{
  return new Table({{
    width: {{ size: 9360, type: WidthType.DXA }},
    columnWidths: [2500, 6860],
    rows: rows.map(([label, value]) => new TableRow({{
      children: [
        new TableCell({{
          borders, width: {{ size: 2500, type: WidthType.DXA }},
          shading: {{ fill: "EBF3FB", type: ShadingType.CLEAR }},
          margins: {{ top: 80, bottom: 80, left: 120, right: 120 }},
          children: [new Paragraph({{ children: [new TextRun({{ text: label, bold: true, font: "Arial", size: 22 }})] }})]
        }}),
        new TableCell({{
          borders, width: {{ size: 6860, type: WidthType.DXA }},
          margins: {{ top: 80, bottom: 80, left: 120, right: 120 }},
          children: [new Paragraph({{ children: [new TextRun({{ text: value || "Not provided", font: "Arial", size: 22 }})] }})]
        }})
      ]
    }}))
  }});
}}

function ddxTable(supporting, against) {{
  const greenBorder = {{ style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" }};
  const greenBorders = {{ top: greenBorder, bottom: greenBorder, left: greenBorder, right: greenBorder }};
  return new Table({{
    width: {{ size: 9360, type: WidthType.DXA }},
    columnWidths: [4680, 4680],
    rows: [
      new TableRow({{
        children: [
          new TableCell({{
            borders: greenBorders, width: {{ size: 4680, type: WidthType.DXA }},
            shading: {{ fill: "EAFAF1", type: ShadingType.CLEAR }},
            margins: {{ top: 80, bottom: 80, left: 120, right: 120 }},
            children: [
              new Paragraph({{ children: [new TextRun({{ text: "Supporting Features", bold: true, font: "Arial", size: 22, color: "1E8449" }})], spacing: {{ after: 80 }} }}),
              ...supporting.map(f => new Paragraph({{ children: [new TextRun({{ text: "• " + f, font: "Arial", size: 20 }})], spacing: {{ after: 40 }} }}))
            ]
          }}),
          new TableCell({{
            borders: greenBorders, width: {{ size: 4680, type: WidthType.DXA }},
            shading: {{ fill: "FDEDEC", type: ShadingType.CLEAR }},
            margins: {{ top: 80, bottom: 80, left: 120, right: 120 }},
            children: [
              new Paragraph({{ children: [new TextRun({{ text: "Arguments Against", bold: true, font: "Arial", size: 22, color: "C0392B" }})], spacing: {{ after: 80 }} }}),
              new Paragraph({{ children: [new TextRun({{ text: against || "None specified.", font: "Arial", size: 20 }})] }})
            ]
          }})
        ]
      }})
    ]
  }});
}}

// ── Data extraction ──────────────────────────────────────────────────────────
const ctx = DATA.clinical_context || {{}};
const meta = DATA.metadata || {{}};
const report = DATA.report || {{}};
const summary = DATA.summary || {{}};
const ddx = DATA.differential_diagnoses || [];
const patches = DATA.patch_importance_ranking || [];
const ihc = DATA.ihc_recommendations || {{}};
const cost = DATA.cost_analysis || {{}};

const topDiag = (summary.top_diagnosis || report.diagnosis || "Unknown").replace(/\\*\\*/g, "");
const confidence = summary.top_diagnosis_confidence || 0;
const reportId = report.report_id || "N/A";
const analysisDate = meta.analysis_timestamp ? meta.analysis_timestamp.split("T")[0] : report.date || "N/A";
const slideName = meta.slide_name || "N/A";

// ── Build document ───────────────────────────────────────────────────────────
const thickBorder = {{ style: BorderStyle.SINGLE, size: 3, color: "2E4A7A" }};
const thickBorders = {{ top: thickBorder, bottom: thickBorder, left: thickBorder, right: thickBorder }};

const patchRows = (DATA.patches || []).map((p, i) => {{
  const imp = patches.find(x => x.patch_id === i);
  const desc = imp ? imp.justification.split("\\n")[0].replace(/\\[.*?\\]/g, "").trim() : "No description available.";
  return new TableRow({{
    children: [
      new TableCell({{
        borders,
        width: {{ size: 1800, type: WidthType.DXA }},
        shading: {{ fill: i === 0 ? "D6EAF8" : i === 1 ? "D5F5E3" : "FDEDEC", type: ShadingType.CLEAR }},
        margins: {{ top: 100, bottom: 100, left: 120, right: 120 }},
        children: [new Paragraph({{ children: [new TextRun({{ text: "Patch " + (i+1), bold: true, font: "Arial", size: 22, color: "1A5276" }})], alignment: AlignmentType.CENTER }})]
      }}),
      new TableCell({{
        borders, width: {{ size: 7560, type: WidthType.DXA }},
        margins: {{ top: 100, bottom: 100, left: 120, right: 120 }},
        children: [new Paragraph({{ children: [new TextRun({{ text: "Coordinates: (" + p.x + ", " + p.y + ")  |  Size: " + p.patch_size + "px  |  Magnification: " + p.magnification + "x", font: "Arial", size: 20, italics: true, color: "888888" }})] }}),
                   new Paragraph({{ children: [new TextRun({{ text: desc.substring(0, 300), font: "Arial", size: 22 }})] }})]
      }})
    ]
  }});
}});

const microscDesc = report.microscopic_description || "";
const microscopicLines = microscDesc.split("\\n").filter(l => l.trim()).map(l =>
  new Paragraph({{ children: [new TextRun({{ text: l.replace(/^- /, ""), font: "Arial", size: 22 }})], spacing: {{ after: 80 }} }})
);

const ddxChildren = [];
ddx.slice(0, 3).forEach((d, idx) => {{
  const diagName = d.diagnosis.replace(/\\*\\*/g, "");
  ddxChildren.push(heading2((idx+1) + ". " + diagName + " — " + d.confidence_score + "% Confidence"));
  ddxChildren.push(ddxTable(d.supporting_features || [], d.arguments_against || ""));
  ddxChildren.push(new Paragraph({{ spacing: {{ after: 160 }} }}));
}});

const ihcPanel = (ihc.recommended_panel || []);
const essentialStains = ihcPanel.filter(s => s.priority === "Essential").map(s => s.stain_name).join(", ") || "None";
const optionalStains = ihcPanel.filter(s => s.priority !== "Essential").map(s => s.stain_name).join(", ") || "None";

const doc = new Document({{
  styles: {{
    default: {{ document: {{ run: {{ font: "Arial", size: 22 }} }} }},
    paragraphStyles: [
      {{ id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: {{ size: 28, bold: true, font: "Arial", color: "2E4A7A" }},
        paragraph: {{ spacing: {{ before: 360, after: 120 }}, outlineLevel: 0 }} }},
      {{ id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: {{ size: 24, bold: true, font: "Arial", color: "1A5276" }},
        paragraph: {{ spacing: {{ before: 240, after: 80 }}, outlineLevel: 1 }} }}
    ]
  }},
  sections: [{{
    properties: {{
      page: {{
        size: {{ width: 12240, height: 15840 }},
        margin: {{ top: 1440, right: 1440, bottom: 1440, left: 1440 }}
      }}
    }},
    headers: {{
      default: new Header({{
        children: [
          new Paragraph({{
            children: [
              new TextRun({{ text: "PATHOLOGY REPORT", bold: true, font: "Arial", size: 24, color: "2E4A7A" }}),
              new TextRun({{ text: "   |   Report ID: " + reportId, font: "Arial", size: 20, color: "666666" }})
            ],
            border: {{ bottom: {{ style: BorderStyle.SINGLE, size: 6, color: "2E4A7A", space: 1 }} }},
            spacing: {{ after: 120 }}
          }})
        ]
      }})
    }},
    footers: {{
      default: new Footer({{
        children: [
          new Paragraph({{
            children: [
              new TextRun({{ text: "Date: " + analysisDate + "   |   Model: " + (meta.model || "N/A") + "   |   Version: " + (meta.version || "N/A") + "   |   Page ", font: "Arial", size: 18, color: "888888" }}),
              new TextRun({{ children: [PageNumber.CURRENT], font: "Arial", size: 18, color: "888888" }})
            ],
            alignment: AlignmentType.CENTER,
            border: {{ top: {{ style: BorderStyle.SINGLE, size: 4, color: "CCCCCC", space: 1 }} }}
          }})
        ]
      }})
    }},
    children: [
      // Title
      new Paragraph({{ children: [new TextRun({{ text: "HISTOPATHOLOGY ANALYSIS REPORT", bold: true, font: "Arial", size: 40, color: "2E4A7A" }})], alignment: AlignmentType.CENTER, spacing: {{ after: 80 }} }}),
      new Paragraph({{ children: [new TextRun({{ text: (ctx.site || "Breast") + " Biopsy — " + (ctx.stain || "H&E"), font: "Arial", size: 24, color: "555555" }})], alignment: AlignmentType.CENTER, spacing: {{ after: 60 }} }}),
      new Paragraph({{ children: [new TextRun({{ text: "AI-Assisted Digital Pathology Review", font: "Arial", size: 22, color: "888888", italics: true }})], alignment: AlignmentType.CENTER, spacing: {{ after: 400 }} }}),

      // Specimen Info
      heading1("SPECIMEN INFORMATION"),
      infoTable([
        ["Report ID", reportId],
        ["Analysis Date", analysisDate],
        ["Slide Name", slideName],
        ["Specimen Type", ctx.specimen_type || report.specimen_type || "Biopsy"],
        ["Site", ctx.site || "Breast"],
        ["Gender", ctx.gender || "Unknown"],
        ["Age", String(ctx.age || "Unknown")],
        ["Stain", ctx.stain || "H&E"],
        ["Total Patches", String(summary.total_patches_analyzed || DATA.patches?.length || 0)],
        ["Clinical History", report.clinical_history || "Not provided"],
      ]),
      new Paragraph({{ spacing: {{ after: 200 }} }}),

      // Primary Diagnosis
      heading1("PRIMARY DIAGNOSIS"),
      new Table({{
        width: {{ size: 9360, type: WidthType.DXA }},
        columnWidths: [9360],
        rows: [new TableRow({{ children: [new TableCell({{
          borders: thickBorders,
          width: {{ size: 9360, type: WidthType.DXA }},
          shading: {{ fill: "EBF5FB", type: ShadingType.CLEAR }},
          margins: {{ top: 200, bottom: 200, left: 240, right: 240 }},
          children: [
            new Paragraph({{ children: [new TextRun({{ text: topDiag, bold: true, font: "Arial", size: 32, color: "1A5276" }})], spacing: {{ after: 80 }} }}),
            new Paragraph({{ children: [new TextRun({{ text: "Confidence Score: ", bold: true, font: "Arial", size: 22 }}), new TextRun({{ text: confidence + "%", bold: true, font: "Arial", size: 22, color: confidence >= 70 ? "1E8449" : "E67E22" }})] }})
          ]
        }})] }})]
      }}),
      new Paragraph({{ spacing: {{ after: 200 }} }}),

      // Patch Overview
      heading1("PATCH OVERVIEW"),
      new Table({{
        width: {{ size: 9360, type: WidthType.DXA }},
        columnWidths: [1800, 7560],
        rows: patchRows.length > 0 ? patchRows : [new TableRow({{ children: [
          new TableCell({{ borders, width: {{ size: 9360, type: WidthType.DXA }}, children: [para("No patch data available.")] }})
        ] }})]
      }}),
      new Paragraph({{ spacing: {{ after: 200 }} }}),

      // Microscopic Description
      heading1("MICROSCOPIC DESCRIPTION"),
      ...(microscopicLines.length > 0 ? microscopicLines : [para(report.microscopic_description || "Not provided.")]),
      new Paragraph({{ spacing: {{ after: 200 }} }}),

      // Differential Diagnoses
      heading1("DIFFERENTIAL DIAGNOSES"),
      ...(ddxChildren.length > 0 ? ddxChildren : [para("No differential diagnoses generated.")]),

      // IHC
      heading1("IMMUNOHISTOCHEMISTRY"),
      infoTable([
        ["Working Diagnosis", (ihc.diagnosis || topDiag).replace(/\\*\\*/g, "")],
        ["Essential Stains", essentialStains],
        ["Optional Stains", optionalStains],
        ["Est. Total Cost", "$" + (ihc.total_cost || 0).toFixed(2)],
        ["Turnaround (hrs)", String(cost.estimated_turnaround_hours || "N/A")],
        ["Requires Approval", cost.requires_approval ? "Yes" : "No"],
        ["Unavailable Stains", (cost.unavailable_stains || []).join(", ") || "None"],
      ]),
      new Paragraph({{ spacing: {{ after: 200 }} }}),

      // Comment / Teaching Point
      heading1("COMMENT & TEACHING POINT"),
      new Table({{
        width: {{ size: 9360, type: WidthType.DXA }},
        columnWidths: [9360],
        rows: [new TableRow({{ children: [new TableCell({{
          borders: {{ top: {{ style: BorderStyle.SINGLE, size: 4, color: "F39C12" }}, bottom: {{ style: BorderStyle.SINGLE, size: 4, color: "F39C12" }}, left: {{ style: BorderStyle.SINGLE, size: 12, color: "F39C12" }}, right: {{ style: BorderStyle.SINGLE, size: 4, color: "F39C12" }} }},
          width: {{ size: 9360, type: WidthType.DXA }},
          shading: {{ fill: "FEF9E7", type: ShadingType.CLEAR }},
          margins: {{ top: 160, bottom: 160, left: 200, right: 200 }},
          children: [
            new Paragraph({{ children: [new TextRun({{ text: "Pathologist Comment", bold: true, font: "Arial", size: 22, color: "D68910" }})], spacing: {{ after: 100 }} }}),
            new Paragraph({{ children: [new TextRun({{ text: (report.comment || "Not provided.").replace(/\\*\\*/g, "").substring(0, 800), font: "Arial", size: 22 }})] }})
          ]
        }})] }})]
      }}),
      new Paragraph({{ spacing: {{ after: 200 }} }}),

      // Summary
      heading1("ANALYSIS SUMMARY"),
      infoTable([
        ["Total Patches Analyzed", String(summary.total_patches_analyzed || 0)],
        ["Top Diagnosis", topDiag],
        ["Confidence Level", confidence + "%"],
        ["Most Important Patch", "Patch " + (summary.most_important_patch || 1)],
        ["IHC Panel Size", String(summary.ihc_panel_size || 0) + " stains"],
        ["Estimated Cost", "$" + (summary.estimated_cost || 0).toFixed(2)],
        ["Model", meta.model || "N/A"],
        ["Report Version", meta.version || "N/A"],
      ]),
      new Paragraph({{ spacing: {{ after: 300 }} }}),

      // Disclaimer
      new Paragraph({{ children: [new TextRun({{ text: "DISCLAIMER", bold: true, font: "Arial", size: 20, color: "888888" }})], spacing: {{ after: 60 }} }}),
      new Paragraph({{ children: [new TextRun({{ text: "This report is generated by an AI-assisted digital pathology system for research and educational purposes only. It does not constitute a clinical diagnosis. All findings must be reviewed by a qualified pathologist in conjunction with complete clinical history, imaging, and laboratory data.", font: "Arial", size: 18, color: "999999", italics: true }})] }})
    ]
  }}]
}});

Packer.toBuffer(doc).then(buf => {{
  fs.writeFileSync("{output_path}", buf);
  console.log("OK");
}});
'''


def generate_report(json_path: str, output_dir: str) -> str:
    """Generate a Word report from a JSON result file."""
    json_path = Path(json_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(json_path) as f:
        data = json.load(f)

    slide_name = data.get("metadata", {}).get("slide_name", json_path.stem)
    # Clean up slide name for use as filename
    safe_name = slide_name.replace(".svs", "").replace(".tif", "")[:60]
    output_path = output_dir / f"{safe_name}_report.docx"

    # Write the JS script
    script_path = Path("/tmp/gen_report_temp.js")
    script = REPORT_SCRIPT_TEMPLATE.format(
        data_json=json.dumps(data, ensure_ascii=False),
        output_path=str(output_path)
    )
    script_path.write_text(script)

    result = subprocess.run(
    ["node", str(script_path)],
    env={**os.environ, "NODE_PATH": str(Path.home() / ".npm-docx/node_modules")},
        capture_output=True, text=True
    )

    if result.returncode != 0 or "OK" not in result.stdout:
        raise RuntimeError(f"Node.js error:\n{result.stderr}\n{result.stdout}")

    return str(output_path)


def main():
    parser = argparse.ArgumentParser(description="Generate Word reports from PathPatch Navigator JSON output.")
    parser.add_argument("json_files", nargs="*", help="JSON result file(s) to process. If omitted, scans tcga_test_output/")
    parser.add_argument("--output", "-o", default="reports_output", help="Output directory for reports (default: reports_output/)")
    parser.add_argument("--input-dir", "-i", default="tcga_test_output", help="Directory to scan for JSON files (default: tcga_test_output/)")
    args = parser.parse_args()

    # Collect JSON files to process
    if args.json_files:
        json_files = [Path(f) for f in args.json_files]
    else:
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            print(f"ERROR: Input directory '{input_dir}' not found.")
            print("Run from your project root, or specify --input-dir path/to/tcga_test_output/")
            sys.exit(1)
        json_files = sorted(input_dir.glob("*_results.json"))
        if not json_files:
            print(f"No *_results.json files found in {input_dir}/")
            sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  PathPatch Navigator — Report Generator")
    print(f"{'='*60}")
    print(f"  Found {len(json_files)} JSON file(s) to process")
    print(f"  Output directory: {args.output}/")
    print(f"{'='*60}\n")

    success = 0
    failed = 0
    for i, json_file in enumerate(json_files, 1):
        print(f"[{i}/{len(json_files)}] Processing: {json_file.name}")
        try:
            output_path = generate_report(str(json_file), args.output)
            print(f"  ✓ Report saved: {output_path}")
            success += 1
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"  Done! {success} report(s) generated, {failed} failed.")
    print(f"  Reports saved to: {Path(args.output).resolve()}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()