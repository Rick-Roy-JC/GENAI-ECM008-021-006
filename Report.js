const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  HeadingLevel, AlignmentType, BorderStyle, WidthType, ShadingType,
  VerticalAlign, PageBreak, LevelFormat, Header, Footer,
  TabStopType, TabStopPosition
} = require('docx');
const fs = require('fs');

// ── Helpers ───────────────────────────────────────────────────────────────
const DARK_BLUE  = "1B3A6B";
const MID_BLUE   = "2E6DA4";
const LIGHT_BLUE = "D5E8F0";
const GREEN      = "1E8449";
const GRAY       = "666666";
const WHITE      = "FFFFFF";
const LIGHT_GRAY = "F2F2F2";

const border = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
const borders = { top: border, bottom: border, left: border, right: border };
const noBorder = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
const noBorders = { top: noBorder, bottom: noBorder, left: noBorder, right: noBorder };

function cell(text, opts = {}) {
  const { bold = false, color = "000000", bg = null, width = 2340, size = 20, align = AlignmentType.LEFT, italic = false } = opts;
  return new TableCell({
    borders,
    width: { size: width, type: WidthType.DXA },
    shading: bg ? { fill: bg, type: ShadingType.CLEAR } : undefined,
    verticalAlign: VerticalAlign.CENTER,
    margins: { top: 80, bottom: 80, left: 120, right: 120 },
    children: [new Paragraph({
      alignment: align,
      children: [new TextRun({ text, bold, color, size, font: "Arial", italics: italic })]
    })]
  });
}

function hcell(text, width = 2340, align = AlignmentType.LEFT) {
  return cell(text, { bold: true, color: WHITE, bg: DARK_BLUE, width, size: 20, align });
}

function body(text, opts = {}) {
  const { bold = false, color = "222222", size = 22, space = 120, italic = false } = opts;
  return new Paragraph({
    spacing: { before: space, after: space },
    children: [new TextRun({ text, bold, color, size, font: "Arial", italics: italic })]
  });
}

function bullet(text) {
  return new Paragraph({
    numbering: { reference: "bullets", level: 0 },
    spacing: { before: 60, after: 60 },
    children: [new TextRun({ text, size: 22, font: "Arial", color: "222222" })]
  });
}

function gap(n = 1) {
  return Array.from({ length: n }, () => new Paragraph({ children: [new TextRun("")] }));
}

function divider() {
  return new Paragraph({
    border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: MID_BLUE, space: 1 } },
    children: [new TextRun("")]
  });
}

// Helper to create page number text (workaround for PageNumber constructor issue)
function getPageNumberText() {
  return new TextRun({ text: "PAGE_NUMBER_PLACEHOLDER", font: "Arial", size: 16, color: GRAY });
}

// ── Document ──────────────────────────────────────────────────────────────
const doc = new Document({
  numbering: {
    config: [
      {
        reference: "bullets",
        levels: [{
          level: 0, format: LevelFormat.BULLET, text: "\u2022",
          alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } }
        }]
      }
    ]
  },
  styles: {
    default: { document: { run: { font: "Arial", size: 22 } } },
    paragraphStyles: [
      {
        id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 32, bold: true, font: "Arial", color: DARK_BLUE },
        paragraph: { spacing: { before: 280, after: 140 }, outlineLevel: 0 }
      },
      {
        id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: true, font: "Arial", color: MID_BLUE },
        paragraph: { spacing: { before: 200, after: 100 }, outlineLevel: 1 }
      },
      {
        id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 22, bold: true, font: "Arial", color: "444444" },
        paragraph: { spacing: { before: 160, after: 80 }, outlineLevel: 2 }
      }
    ]
  },
  sections: [{
    properties: {
      page: {
        size: { width: 11906, height: 16838 },
        margin: { top: 1134, right: 1134, bottom: 1134, left: 1134 }
      }
    },
    headers: {
      default: new Header({
        children: [
          new Paragraph({
            border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: MID_BLUE, space: 1 } },
            children: [
              new TextRun({ text: "CS5202 GenAI and LLM  |  Spring 2026  |  Final Report", font: "Arial", size: 18, color: GRAY }),
              new TextRun({ text: "         Team 2 — GENAI-ECM008-021-006", font: "Arial", size: 18, color: GRAY })
            ]
          })
        ]
      })
    },
    footers: {
      default: new Footer({
        children: [
          new Paragraph({
            border: { top: { style: BorderStyle.SINGLE, size: 4, color: MID_BLUE, space: 1 } },
            tabStops: [{ type: TabStopType.RIGHT, position: 9026 }],
            children: [
              new TextRun({ text: "Reliable Clinical NLP using RAG with Continuous Knowledge Updating", font: "Arial", size: 16, color: GRAY }),
              new TextRun({ text: "\tPage ", font: "Arial", size: 16, color: GRAY }),
              getPageNumberText()
            ]
          })
        ]
      })
    },
    children: [

      // ══════════════════════════════════════════════════════
      // TITLE PAGE
      // ══════════════════════════════════════════════════════

      new Table({
        width: { size: 9026, type: WidthType.DXA },
        columnWidths: [9026],
        rows: [
          new TableRow({ children: [
            new TableCell({
              borders: noBorders,
              width: { size: 9026, type: WidthType.DXA },
              shading: { fill: DARK_BLUE, type: ShadingType.CLEAR },
              margins: { top: 280, bottom: 280, left: 280, right: 280 },
              children: [
                new Paragraph({ alignment: AlignmentType.CENTER, children: [
                  new TextRun({ text: "CS5202 — Generative AI and LLM", font: "Arial", size: 28, bold: true, color: WHITE })
                ]}),
                new Paragraph({ alignment: AlignmentType.CENTER, children: [
                  new TextRun({ text: "Spring 2026  |  Final Report", font: "Arial", size: 22, color: "AACCEE" })
                ]}),
              ]
            })
          ]})
        ]
      }),

      ...gap(1),

      new Table({
        width: { size: 9026, type: WidthType.DXA },
        columnWidths: [9026],
        rows: [
          new TableRow({ children: [
            new TableCell({
              borders: noBorders,
              width: { size: 9026, type: WidthType.DXA },
              shading: { fill: MID_BLUE, type: ShadingType.CLEAR },
              margins: { top: 200, bottom: 200, left: 280, right: 280 },
              children: [
                new Paragraph({ alignment: AlignmentType.CENTER, children: [
                  new TextRun({ text: "Reliable Clinical NLP using Retrieval-Augmented LLMs", font: "Arial", size: 30, bold: true, color: WHITE })
                ]}),
                new Paragraph({ alignment: AlignmentType.CENTER, children: [
                  new TextRun({ text: "with Continuous Knowledge Updating", font: "Arial", size: 28, bold: true, color: WHITE })
                ]}),
              ]
            })
          ]})
        ]
      }),

      ...gap(1),

      new Table({
        width: { size: 9026, type: WidthType.DXA },
        columnWidths: [2600, 6426],
        rows: [
          new TableRow({ children: [hcell("Course", 2600), cell("CS5202 — GenAI and LLM, Spring 2026", { width: 6426 })] }),
          new TableRow({ children: [hcell("Instructor", 2600), cell("Prof. Nidhi Goyal", { width: 6426 })] }),
          new TableRow({ children: [hcell("Domain", 2600), cell("Medical GenAI (Domain E)", { width: 6426 })] }),
          new TableRow({ children: [hcell("Project", 2600), cell("Project 19/20  |  Team No. 2", { width: 6426 })] }),
          new TableRow({ children: [hcell("Repository", 2600), cell("github.com/Rick-Roy-JC/GENAI-ECM008-021-006", { width: 6426 })] }),
          new TableRow({ children: [hcell("Final Evaluation", 2600), cell("May 15, 2026", { width: 6426 })] }),
        ]
      }),

      ...gap(1),

       new Table({
        width: { size: 9026, type: WidthType.DXA },
        columnWidths: [3009, 3009, 3008],
        rows: [
          new TableRow({ children: [hcell("Name", 3009), hcell("Roll Number", 3009), hcell("Role", 3008)] }),
          new TableRow({ children: [cell("Aritra Roy", { width: 3009 }), cell("SE23UECM008", { width: 3009 }), cell("Team Lead / ML Pipeline", { width: 3008 })] }),
          new TableRow({ children: [cell("Dheeraj Reddy", { width: 3009 }), cell("SE23UECM021", { width: 3009 }), cell("Data Engineering / RAG", { width: 3008 })] }),
          new TableRow({ children: [cell("A. Sai Praneeth", { width: 3009 }), cell("SE23UECM006", { width: 3009 }), cell("Evaluation / Knowledge Update", { width: 3008 })] }),
          new TableRow({ children: [cell("Ruthwik Reddy", { width: 3009 }), cell("SE23UCSE085", { width: 3009 }), cell("Frontend / Integration", { width: 3008 })] }),
        ]
      }),

      ...gap(1),
      new Paragraph({ children: [new PageBreak()] }),

      // SECTION 1 — PROBLEM AND MOTIVATION
      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun({ text: "1.  Problem and Motivation", font: "Arial", bold: true, color: DARK_BLUE, size: 32 })] }),
      divider(),

      body("Clinical decision-making requires accurate and continuously updated medical knowledge. Modern Large Language Models (LLMs), despite their remarkable capabilities, suffer from two critical limitations: static training data that becomes outdated as new research is published, and hallucination — generating plausible but factually incorrect information. In healthcare settings, such errors directly threaten patient safety and treatment outcomes."),
      body("Medical knowledge evolves rapidly. New drug approvals, revised treatment guidelines, and emerging infectious disease protocols cannot be reflected in a model whose weights are frozen at training time. For example, aspirin recommendations for primary prevention changed significantly in 2025 — a static LLM trained before this update would continue providing outdated advice to clinicians."),

      new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun({ text: "1.1  Who Is Affected", font: "Arial", bold: true, color: MID_BLUE, size: 26 })] }),
      bullet("Healthcare providers who rely on clinical decision support for trusted, up-to-date guidance"),
      bullet("Patients, especially in resource-limited settings where expert consultation is unavailable"),
      bullet("Healthcare organisations that face liability and loss of trust when AI systems produce outdated or incorrect clinical advice"),

      new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun({ text: "1.2  Why Existing Solutions Fall Short", font: "Arial", bold: true, color: MID_BLUE, size: 26 })] }),

      new Table({
        width: { size: 9026, type: WidthType.DXA },
        columnWidths: [2800, 6226],
        rows: [
          new TableRow({ children: [hcell("Limitation", 2800), hcell("Description", 6226)] }),
          new TableRow({ children: [cell("Knowledge Staleness", { width: 2800, bold: true, color: MID_BLUE }), cell("Static LLMs cannot incorporate new guidelines or drug approvals published after training cutoff.", { width: 6226 })] }),
          new TableRow({ children: [cell("Hallucination", { width: 2800, bold: true, color: MID_BLUE }), cell("Without retrieval grounding, generated outputs are often factually incorrect and overconfident.", { width: 6226 })] }),
          new TableRow({ children: [cell("Static RAG", { width: 2800, bold: true, color: MID_BLUE }), cell("Standard RAG improves factual grounding but relies on a fixed knowledge base that is never updated.", { width: 6226 })] }),
        ]
      }),

      ...gap(1),
      new Paragraph({ children: [new PageBreak()] }),

      // SECTION 2 — METHOD
      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun({ text: "2.  Method", font: "Arial", bold: true, color: DARK_BLUE, size: 32 })] }),
      divider(),

      body("We propose a Retrieval-Augmented Generation (RAG) system for reliable clinical NLP with a continuous knowledge-updating mechanism. The system has three core components:"),

      new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun({ text: "2.1  System Components", font: "Arial", bold: true, color: MID_BLUE, size: 26 })] }),

      new Table({
        width: { size: 9026, type: WidthType.DXA },
        columnWidths: [2600, 4226, 2200],
        rows: [
          new TableRow({ children: [hcell("Component", 2600), hcell("Description", 4226), hcell("Technology", 2200)] }),
          new TableRow({ children: [
            cell("Dense Retrieval", { width: 2600, bold: true, color: MID_BLUE }),
            cell("Clinical text encoded into 768-dim dense vectors and indexed for fast cosine similarity search.", { width: 4226 }),
            cell("PubMedBERT + FAISS", { width: 2200, italic: true })
          ]}),
          new TableRow({ children: [
            cell("LLM Generation", { width: 2600, bold: true, color: MID_BLUE }),
            cell("Top-3 retrieved passages concatenated as context and fed to an instruction-tuned LLM to generate a grounded answer.", { width: 4226 }),
            cell("Flan-T5-Base", { width: 2200, italic: true })
          ]}),
          new TableRow({ children: [
            cell("Continuous Update", { width: 2600, bold: true, color: GREEN }),
            cell("New clinical documents embedded and added to the live FAISS index incrementally — no full rebuild required. Every update logged with timestamp and source.", { width: 4226 }),
            cell("FAISS index.add() + Update Log", { width: 2200, italic: true })
          ]}),
        ]
      }),

      ...gap(1),

      new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun({ text: "2.2  Pipeline", font: "Arial", bold: true, color: MID_BLUE, size: 26 })] }),

      new Table({
        width: { size: 9026, type: WidthType.DXA },
        columnWidths: [500, 2500, 6026],
        rows: [
          new TableRow({ children: [hcell("#", 500), hcell("Stage", 2500), hcell("Description", 6026)] }),
          new TableRow({ children: [cell("1", { width: 500, align: AlignmentType.CENTER }), cell("Data Loading", { width: 2500, bold: true }), cell("PubMedQA loaded from HuggingFace. 1000 samples across train/val/test (800/100/100).", { width: 6026 })] }),
          new TableRow({ children: [cell("2", { width: 500, align: AlignmentType.CENTER }), cell("Text Chunking", { width: 2500, bold: true }), cell("Context passages split into 200-word overlapping chunks (40-word overlap) to preserve sentence boundaries.", { width: 6026 })] }),
          new TableRow({ children: [cell("3", { width: 500, align: AlignmentType.CENTER }), cell("Embedding", { width: 2500, bold: true }), cell("Each chunk encoded using pritamdeka/S-PubMedBert-MS-MARCO into 768-dimensional vectors. Biomedical vocabulary improves clinical retrieval.", { width: 6026 })] }),
          new TableRow({ children: [cell("4", { width: 500, align: AlignmentType.CENTER }), cell("FAISS Indexing", { width: 2500, bold: true }), cell("Embeddings stored in FAISS IndexFlatIP. Inner product search over normalised vectors equals cosine similarity. Index size: 1810 vectors.", { width: 6026 })] }),
          new TableRow({ children: [cell("5", { width: 500, align: AlignmentType.CENTER }), cell("Retrieval", { width: 2500, bold: true }), cell("Query embedded and top-3 most similar passages retrieved from the index (mean top-1 similarity score: 0.94).", { width: 6026 })] }),
          new TableRow({ children: [cell("6", { width: 500, align: AlignmentType.CENTER }), cell("Generation", { width: 2500, bold: true }), cell("Retrieved passages concatenated into a structured prompt. Flan-T5-Base generates a yes/no/maybe answer using beam search (num_beams=4, max_new_tokens=5).", { width: 6026 })] }),
          new TableRow({ children: [
            cell("7", { width: 500, align: AlignmentType.CENTER, color: WHITE, bg: GREEN }),
            cell("Knowledge Update", { width: 2500, bold: true, color: GREEN }),
            cell("New documents embedded and added via FAISS index.add() — O(n) operation, no rebuild. Every update logged to update_log.jsonl with timestamp, source, and chunk count.", { width: 6026 })
          ]}),
        ]
      }),

      ...gap(1),

      new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun({ text: "2.3  Continuous Knowledge Updating — Core Contribution", font: "Arial", bold: true, color: MID_BLUE, size: 26 })] }),

      body("The key novelty of this system is the continuous knowledge update mechanism. Unlike standard RAG systems that require a full index rebuild when new information arrives, our system supports incremental updates:"),
      bullet("New clinical documents (WHO guidelines, drug approvals, research papers) are chunked and embedded using the same biomedical encoder"),
      bullet("Embeddings are added directly to the live FAISS index using index.add() — the operation is near-instantaneous (<1 second per document)"),
      bullet("Passage metadata is appended to the passage store with a timestamp, source identifier, and update tag"),
      bullet("Every update is logged to results/update_log.jsonl for full reproducibility and audit trail"),
      bullet("The updated index is persisted to disk so knowledge additions survive across sessions"),

      ...gap(1),
      new Paragraph({ children: [new PageBreak()] }),

      // SECTION 3 — EXPERIMENTS AND RESULTS
      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun({ text: "3.  Experiments and Results", font: "Arial", bold: true, color: DARK_BLUE, size: 32 })] }),
      divider(),

      new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun({ text: "3.1  Experimental Setup", font: "Arial", bold: true, color: MID_BLUE, size: 26 })] }),

      new Table({
        width: { size: 9026, type: WidthType.DXA },
        columnWidths: [3013, 3013, 3000],
        rows: [
          new TableRow({ children: [hcell("Component", 3013), hcell("Choice", 3013), hcell("Reason", 3000)] }),
          new TableRow({ children: [cell("Dataset", { width: 3013 }), cell("PubMedQA (pqa_labeled)", { width: 3013 }), cell("Biomedical Q&A with yes/no/maybe labels", { width: 3000 })] }),
          new TableRow({ children: [cell("Embedding Model", { width: 3013 }), cell("S-PubMedBert-MS-MARCO", { width: 3013 }), cell("Biomedical domain, 768-dim", { width: 3000 })] }),
          new TableRow({ children: [cell("LLM", { width: 3013 }), cell("google/flan-t5-base", { width: 3013 }), cell("Instruction-tuned, ~250M params, local inference", { width: 3000 })] }),
          new TableRow({ children: [cell("Index Type", { width: 3013 }), cell("FAISS IndexFlatIP", { width: 3013 }), cell("Supports incremental addition", { width: 3000 })] }),
          new TableRow({ children: [cell("Top-K", { width: 3013 }), cell("3", { width: 3013 }), cell("Balance between context and prompt length", { width: 3000 })] }),
          new TableRow({ children: [cell("Eval Samples", { width: 3013 }), cell("100 (test split)", { width: 3013 }), cell("Held-out from index construction", { width: 3000 })] }),
        ]
      }),

      ...gap(1),

      new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun({ text: "3.2  Main Evaluation Results", font: "Arial", bold: true, color: MID_BLUE, size: 26 })] }),

      body("Table 2 shows the full quantitative evaluation on 100 held-out PubMedQA test samples."),

      new Table({
        width: { size: 9026, type: WidthType.DXA },
        columnWidths: [3626, 2700, 2700],
        rows: [
          new TableRow({ children: [hcell("Metric", 3626), hcell("Value", 2700, AlignmentType.CENTER), hcell("Notes", 2700)] }),
          new TableRow({ children: [cell("Exact Match Accuracy", { width: 3626 }), cell("37.0%", { width: 2700, align: AlignmentType.CENTER, bold: true }), cell("Above random baseline of 33.3%", { width: 2700 })] }),
          new TableRow({ children: [cell("Yes F1", { width: 3626 }), cell("0.4356", { width: 2700, align: AlignmentType.CENTER }), cell("Strongest class performance", { width: 2700 })] }),
          new TableRow({ children: [cell("No F1", { width: 3626 }), cell("0.3659", { width: 2700, align: AlignmentType.CENTER }), cell("Moderate performance", { width: 2700 })] }),
          new TableRow({ children: [cell("Maybe F1", { width: 3626 }), cell("0.0000", { width: 2700, align: AlignmentType.CENTER, color: "CC0000" }), cell("Model never predicts maybe — known Flan-T5 limitation", { width: 2700 })] }),
          new TableRow({ children: [cell("Macro F1", { width: 3626 }), cell("0.2672", { width: 2700, align: AlignmentType.CENTER }), cell("Dragged down by zero maybe F1", { width: 2700 })] }),
          new TableRow({ children: [cell("Mean Top-1 Retrieval Score", { width: 3626 }), cell("0.9421", { width: 2700, align: AlignmentType.CENTER, bold: true, color: GREEN }), cell("Excellent — PubMedBERT working well", { width: 2700 })] }),
          new TableRow({ children: [cell("Mean Top-3 Retrieval Score", { width: 3626 }), cell("0.9209", { width: 2700, align: AlignmentType.CENTER }), cell("Consistently high across all queries", { width: 2700 })] }),
        ]
      }),

      ...gap(1),

      new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun({ text: "3.3  Ablation Study", font: "Arial", bold: true, color: MID_BLUE, size: 26 })] }),

      body("We compare three system configurations to isolate the contribution of each component:"),

      new Table({
        width: { size: 9026, type: WidthType.DXA },
        columnWidths: [3026, 1500, 1500, 1500, 1500],
        rows: [
          new TableRow({ children: [
            hcell("Metric", 3026),
            hcell("No RAG", 1500, AlignmentType.CENTER),
            hcell("RAG Static", 1500, AlignmentType.CENTER),
            hcell("RAG+Update", 1500, AlignmentType.CENTER),
            hcell("Best", 1500, AlignmentType.CENTER),
          ]}),
          new TableRow({ children: [
            cell("Accuracy (%)", { width: 3026 }),
            cell("37.0", { width: 1500, align: AlignmentType.CENTER }),
            cell("37.0", { width: 1500, align: AlignmentType.CENTER }),
            cell("36.0", { width: 1500, align: AlignmentType.CENTER }),
            cell("No RAG / RAG", { width: 1500, align: AlignmentType.CENTER }),
          ]}),
          new TableRow({ children: [
            cell("Macro F1", { width: 3026 }),
            cell("0.2459", { width: 1500, align: AlignmentType.CENTER }),
            cell("0.2672", { width: 1500, align: AlignmentType.CENTER, bold: true, color: GREEN }),
            cell("0.2590", { width: 1500, align: AlignmentType.CENTER }),
            cell("RAG Static", { width: 1500, align: AlignmentType.CENTER }),
          ]}),
          new TableRow({ children: [
            cell("Yes F1", { width: 3026 }),
            cell("0.2462", { width: 1500, align: AlignmentType.CENTER }),
            cell("0.4356", { width: 1500, align: AlignmentType.CENTER, bold: true, color: GREEN }),
            cell("0.4314", { width: 1500, align: AlignmentType.CENTER }),
            cell("RAG Static", { width: 1500, align: AlignmentType.CENTER }),
          ]}),
          new TableRow({ children: [
            cell("No F1", { width: 3026 }),
            cell("0.4915", { width: 1500, align: AlignmentType.CENTER, bold: true, color: GREEN }),
            cell("0.3659", { width: 1500, align: AlignmentType.CENTER }),
            cell("0.3457", { width: 1500, align: AlignmentType.CENTER }),
            cell("No RAG", { width: 1500, align: AlignmentType.CENTER }),
          ]}),
          new TableRow({ children: [
            cell("Maybe F1", { width: 3026 }),
            cell("0.00", { width: 1500, align: AlignmentType.CENTER }),
            cell("0.00", { width: 1500, align: AlignmentType.CENTER }),
            cell("0.00", { width: 1500, align: AlignmentType.CENTER }),
            cell("—", { width: 1500, align: AlignmentType.CENTER }),
          ]}),
        ]
      }),

      ...gap(1),
      body("Key finding: RAG retrieval improves Yes F1 by 77% over No RAG (0.2462 → 0.4356), demonstrating that retrieved context genuinely helps the model identify positive clinical evidence. Overall accuracy remains flat across conditions — this is attributed to Flan-T5-Base being insufficiently sensitive to retrieved context, a known limitation of small instruction-tuned models discussed in Section 4.", { color: "333333" }),

      ...gap(1),

      new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun({ text: "3.4  Continuous Knowledge Update Experiment", font: "Arial", bold: true, color: MID_BLUE, size: 26 })] }),

      body("We demonstrate the continuous update mechanism using a focused experiment with a pre-built clinical knowledge base of 8 documents, queried before and after adding 3 new 2025 medical guidelines."),

      new Table({
        width: { size: 9026, type: WidthType.DXA },
        columnWidths: [2800, 1556, 1557, 1557, 1556],
        rows: [
          new TableRow({ children: [
            hcell("Query Topic", 2800),
            hcell("Before Source", 1556),
            hcell("After Source", 1557),
            hcell("Src Changed?", 1557),
            hcell("Ans Changed?", 1556),
          ]}),
          new TableRow({ children: [
            cell("Aspirin primary prevention", { width: 2800 }),
            cell("clinical_kb_aspirin_old", { width: 1556, italic: true, size: 18 }),
            cell("clinical_kb_aspirin_old", { width: 1557, italic: true, size: 18 }),
            cell("No", { width: 1557, align: AlignmentType.CENTER }),
            cell("No", { width: 1556, align: AlignmentType.CENTER }),
          ]}),
          new TableRow({ children: [
            cell("Metformin in kidney disease", { width: 2800 }),
            cell("clinical_kb_metformin_old", { width: 1556, italic: true, size: 18 }),
            cell("FDA_metformin_2025", { width: 1557, italic: true, size: 18, color: GREEN }),
            cell("YES", { width: 1557, align: AlignmentType.CENTER, bold: true, color: GREEN }),
            cell("No", { width: 1556, align: AlignmentType.CENTER }),
          ]}),
          new TableRow({ children: [
            cell("Blood pressure target", { width: 2800 }),
            cell("clinical_kb_bp_old", { width: 1556, italic: true, size: 18 }),
            cell("NEJM_bp_target_2025", { width: 1557, italic: true, size: 18, color: GREEN }),
            cell("YES", { width: 1557, align: AlignmentType.CENTER, bold: true, color: GREEN }),
            cell("No", { width: 1556, align: AlignmentType.CENTER }),
          ]}),
        ]
      }),

      ...gap(1),
      body("Index size grew from 8 to 11 vectors. Update latency was under 1 second per document with no index rebuild required. Retrieval correctly prioritised newly added guidelines for 2 out of 3 queries, demonstrating that the update mechanism functions as designed."),

      ...gap(1),
      new Paragraph({ children: [new PageBreak()] }),

      // SECTION 4 — ANALYSIS
      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun({ text: "4.  Analysis", font: "Arial", bold: true, color: DARK_BLUE, size: 32 })] }),
      divider(),

      new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun({ text: "4.1  What Worked", font: "Arial", bold: true, color: MID_BLUE, size: 26 })] }),

      bullet("The full data pipeline ran end-to-end without errors — PubMedQA loaded, cleaned, chunked, embedded, and indexed reliably"),
      bullet("PubMedBERT retrieval is strong, achieving mean top-1 similarity scores of 0.94, significantly higher than the general MiniLM baseline"),
      bullet("RAG improves Yes F1 by 77% over No RAG — retrieved context genuinely helps identify positive clinical evidence"),
      bullet("The continuous knowledge update mechanism correctly incorporated new guidelines: retrieval source changed to newly added documents for 2/3 test queries immediately after update, with under 1 second update latency and no index rebuild"),
      bullet("Prompt engineering improvement alone drove a 4x accuracy increase from Milestone 1 (10%) to the final evaluation (37%-40%), demonstrating the importance of instruction format for small LLMs"),
      bullet("Update logging provides a complete audit trail of knowledge changes, which is essential for clinical deployment"),

      new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun({ text: "4.2  Limitations and Honest Analysis", font: "Arial", bold: true, color: MID_BLUE, size: 26 })] }),

      bullet("Flan-T5-Base never predicts 'maybe', resulting in zero F1 for that class. Small instruction-tuned models are overconfident and avoid uncertainty. A larger model (Flan-T5-Large or BioGPT) would likely produce better-calibrated outputs"),
      bullet("Overall accuracy (37%) remains flat across ablation conditions. This is consistent with published findings on small LLMs: models with fewer than 500M parameters tend to generate answers from pretrained weights regardless of retrieved context. The retrieval component works correctly — the bottleneck is generation sensitivity"),
      bullet("The knowledge update experiment used simulated documents rather than real PubMed API data. In production deployment, a PubMed RSS feed or WHO bulletin webhook would automate real-time updates"),
      bullet("ROUGE-L is near zero because we compare single-word answers (yes/no/maybe) against long reference answers. This metric is not appropriate for this task and would be replaced with token-level F1 in future work"),
      bullet("The aspirin query did not change source after the WHO 2025 update was added, likely because the old document had a higher cosine similarity to the query — this reveals a retrieval ranking limitation that re-ranking techniques could address"),

      new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun({ text: "4.3  Does the System Address the Stated Problem?", font: "Arial", bold: true, color: MID_BLUE, size: 26 })] }),

      body("Partially. The retrieval component is functioning well — PubMedBERT successfully grounds answers in relevant clinical passages, and the continuous update mechanism correctly prioritises newly added knowledge. The generation component is limited by model size, preventing full exploitation of retrieved context. The core claim of the project — that a RAG system can be updated with new clinical knowledge without rebuilding — is demonstrated and verified. A production system would require a larger generation model to fully realise the benefits."),

      ...gap(1),
      new Paragraph({ children: [new PageBreak()] }),

      // SECTION 5 — REFERENCES
      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun({ text: "5.  References", font: "Arial", bold: true, color: DARK_BLUE, size: 32 })] }),
      divider(),

      body("[1] Lewis, P., Perez, E., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS 2020."),
      body("[2] Singhal, K., Azizi, S., et al. (2023). Large Language Models Encode Clinical Knowledge. Nature, 620, 172–180. (Med-PaLM)"),
      body("[3] Jin, Q., et al. (2024). MedDec: A Dataset for Extracting Medical Decisions from Discharge Summaries. ACL Findings 2024."),
      body("[4] Gu, Y., Tinn, R., et al. (2021). Domain-Specific Language Model Pretraining for Biomedical NLP. ACM CHIL 2021. (PubMedBERT)"),
      body("[5] Jin, Q., Dhingra, B., et al. (2019). PubMedQA: A Biomedical Research Question Answering Dataset. EMNLP 2019."),
      body("[6] Johnson, A.E.W., et al. (2016). MIMIC-III, a freely accessible critical care database. Scientific Data, 3, 160035."),

    ]
  }]
});

// Create results folder if it doesn't exist
if (!fs.existsSync('./results')) {
  fs.mkdirSync('./results');
}

const reportPath = './results/Final_Report_GENAI-ECM008-021-006.docx';
Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync(reportPath, buffer);
  console.log(`Report written successfully to ${reportPath}`);
});