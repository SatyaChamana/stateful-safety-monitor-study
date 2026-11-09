// Manuscript builder. Usage: node build.js <figures_dir> <out.docx>
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, Table, TableRow, TableCell,
  WidthType, ShadingType, BorderStyle, ImageRun, Footer, Header, PageNumber, LevelFormat,
  PageBreak,
} = require("docx");

const FIG = process.argv[2];
const OUT = process.argv[3];
const FONT = "Times New Roman";
const W = 9360; // content width, DXA (US Letter, 1in margins)

// ---------- helpers ----------
function runs(text, base = {}) {
  // **bold**, *italic*
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*)/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), ...base }));
    const t = m[0];
    if (t.startsWith("**")) out.push(new TextRun({ text: t.slice(2, -2), bold: true, ...base }));
    else out.push(new TextRun({ text: t.slice(1, -1), italics: true, ...base }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), ...base }));
  return out;
}
const P = (text, opts = {}) => new Paragraph({ children: runs(text), spacing: { after: 120 },
  alignment: AlignmentType.JUSTIFIED, ...opts });
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)] });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] });
const H3 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun(t)] });
const B = (text, level = 0) => new Paragraph({ children: runs(text), numbering: { reference: "bullets", level },
  spacing: { after: 60 } });
const N = (text) => new Paragraph({ children: runs(text), numbering: { reference: "numbers", level: 0 },
  spacing: { after: 60 } });
const caption = (text) => new Paragraph({ children: runs(text, { size: 19 }), spacing: { before: 60, after: 200 },
  alignment: AlignmentType.LEFT });

const border = { style: BorderStyle.SINGLE, size: 4, color: "BFBFBF" };
const borders = { top: border, bottom: border, left: border, right: border };
function table(headers, rows, widths, title) {
  const total = widths.reduce((a, b) => a + b, 0);
  if (total !== W) widths = widths.map((w) => Math.round((w * W) / total));
  const diff = W - widths.reduce((a, b) => a + b, 0);
  widths[widths.length - 1] += diff;
  const cell = (text, i, head) => new TableCell({
    borders, width: { size: widths[i], type: WidthType.DXA },
    shading: head ? { fill: "E8EEF6", type: ShadingType.CLEAR, color: "auto" } : undefined,
    margins: { top: 50, bottom: 50, left: 80, right: 80 },
    children: [new Paragraph({ children: runs(String(text), { size: 17, bold: head || undefined }),
      alignment: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER })],
  });
  const t = new Table({
    width: { size: W, type: WidthType.DXA }, columnWidths: widths,
    rows: [new TableRow({ tableHeader: true, children: headers.map((h, i) => cell(h, i, true)) }),
      ...rows.map((r) => new TableRow({ children: r.map((c, i) => cell(c, i, false)) }))],
  });
  const out = [];
  if (title) out.push(new Paragraph({ children: runs(title, { size: 19 }), spacing: { before: 160, after: 80 },
    keepNext: true }));
  out.push(t, new Paragraph({ children: [], spacing: { after: 120 } }));
  return out;
}
function figure(file, w, h, cap) {
  const data = fs.readFileSync(path.join(FIG, file));
  const width = 600, height = Math.round((600 * h) / w);
  return [new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 120 },
    children: [new ImageRun({ type: "png", data, transformation: { width, height },
      altText: { title: file, description: cap, name: file } })] }), caption(cap)];
}

// ---------- content ----------
const body = [];
const add = (...xs) => xs.flat().forEach((x) => body.push(x));

// Title block
add(
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 600, after: 240 }, children: [
    new TextRun({ text: "How Much History Does a Sequential Misuse Monitor Need?", bold: true, size: 34, font: FONT })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 360 }, children: [
    new TextRun({ text: "A reproduction and a controlled comparison of history selection on DecomposedHarm", size: 26, font: FONT })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80 }, children: [new TextRun({ text: "Satya Chamana", size: 24 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80 }, children: [
    new TextRun({ text: "Independent research study. Compute: Clipper HPC cluster, Grand Valley State University.", size: 20, italics: true })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 480 }, children: [
    new TextRun({ text: "Manuscript draft, not peer reviewed. Version phase3-v1.1, revised 8 November 2026 (first version 3 November 2026).", size: 20, italics: true })] }),
);

add(H1("Abstract"));
add(P("Decomposition attacks split a harmful goal into steps that each look harmless, so a monitor that judges one request at a time misses them. Sequential monitors judge each new request together with earlier ones, but how much earlier context they need, and how it should be chosen, is open. We first reproduce a published lightweight sequential monitor (Llama-3.1-8B-Instruct on DecomposedHarm): under the authors' scoring rule, our F1 matches their released outputs to within 0.001 for both prompts with saved outputs. We then hold the judge fixed and vary only the history it sees, across ten conditions specified before testing, with thresholds calibrated on validation to a 5% false-alarm budget and frozen before a single test run on 528 sequences. **Top-3 retrieval is non-inferior to full history** in timely recall for all three retrievers (every lower 95% bound at or above −0.011, against a margin of 0.05 fixed before the test run), with no increase in false alarms detected. Because the sequences are short, top-3 input equals full history in many of the decisions that matter, so this result is specific to the benchmark. Full history detects more attacks in time than the current request alone (+0.114, 95% CI +0.068 to +0.156) but also flags more benign sequences (+0.049). In exploratory analyses, the overall gain disappears at equal false-alarm counts, history helps mainly on QA and agent tasks, and at the frozen thresholds it raised false alarms on benign agent and image tasks. Retrieval did not beat the three most recent steps; the two saw identical inputs in roughly half of the decisions that decide timely detection."));
add(P("**Keywords:** AI safety; misuse monitoring; decomposition attacks; stateful monitoring; retrieval; reproducibility; reproduction."));


// 1 Introduction
add(H1("1. Introduction"));
add(P("A language-model safety filter that looks at one request at a time can be defeated by splitting a harmful task into pieces that are each acceptable on their own. Multi-turn escalation [Russinovich et al., 2024], leakage of harmful information through individually safe answers [Glukhov et al., 2024], and the misuse of combinations of safe models [Jones et al., 2024] are all versions of this problem. Chen et al. [2025] introduced DecomposedHarm, a benchmark of harmful tasks split into sequences of steps with matched benign sequences, and showed that a lightweight model monitoring the *cumulative* sequence detects many such attacks."));
add(P("A deployed monitor cannot always see everything that came before. Context windows, cost, latency, retention policies and privacy rules all limit how much history it may keep and read. That raises a design question that a cumulative monitor does not answer: **which earlier steps should the monitor see, and how many?** If a small, well-chosen subset of history is enough, a monitor can read less at each decision; only a selector that caps what is stored, such as recency, also lets it keep less user data, since retrieval must keep the full history to search it. If a small subset is not enough, the reasons matter: the right evidence may not be retrieved, or the judge may misread it when it is."));
add(P("This study takes two steps toward that question."));
add(N("**Reproduction (Phase 2).** We re-implement the published monitor protocol and run three of its prompts with a local Llama-3.1-8B-Instruct, comparing against both the paper's tables and the authors' released per-run outputs."));
add(N("**Controlled history-selection study (Phase 3).** With one fixed judge and prompt, we vary only which earlier steps the judge sees, across ten conditions specified before testing. Thresholds are calibrated per condition on validation, frozen, and evaluated once on a held-out test set."));
add(P("Our contributions are: (i) an independent reproduction that matches the authors' released outputs closely; (ii) evidence that top-3 retrieval matches full history in timely detection on this benchmark, where top-3 often equals full history; (iii) an exploratory breakdown of where history helps (decomposed question answering) and where it costs false alarms (benign agent and image tasks); and (iv) a transparent account of a calibration failure and the pre-test amendment that fixed it."));

// 2 Background
add(H1("2. Background"));
add(H2("2.1 Sequential monitoring of decomposition attacks"));
add(P("In DecomposedHarm, each harmful sequence carries a *harm index*: the step at which harm would occur if the model complied with every step up to and including it [Chen et al., 2025]. A monitor that flags a harmful sequence at or before that step has detected it in time; flagging later is too late. The published monitor scores the cumulative prefix at each step and fires on the first step whose score passes a threshold chosen on validation data."));
add(P("The harm index is built differently per task type, and the difference matters for interpretation. For question-answering (QA) sequences it is the shortest prefix at which a reference model's answers were judged harmful by StrongREJECT [Souly et al., 2024]; for image tasks it is always the last step; for agent tasks it was annotated by a language model. We treat these as given labels and do not re-annotate them."));
add(H2("2.2 History selection as retrieval"));
add(P("Choosing which earlier steps to show the judge is a retrieval problem in which the current request is the query and earlier steps are the documents. We compare a recency baseline, lexical retrieval with BM25 [Robertson and Zaragoza, 2009], a general pretrained sentence encoder [Reimers and Gurevych, 2019; Wang et al., 2020], and their reciprocal-rank fusion [Cormack et al., 2009]. Selected steps are always presented to the judge in their original order. Monitoring with limited, selected context also connects to the broader control-and-monitoring literature [Greenblatt et al., 2023]."));

// 3 Methods
add(H1("3. Methods"));
add(H2("3.1 Data"));
add(P("We use DecomposedHarm from Hugging Face (YuehHanChen/DecomposedHarm, revision 25e1d59), accessed under the authors' evaluation-only terms; we never train on it. The authors' code is pinned at commit 79f5b41. Each row has an identifier, an original goal, the decomposition into steps, the harm index (1-based, inclusive, verified against the upstream code), a label and a task type."));
add(P("The paper evaluates on a balanced subset built by an unseeded shuffle, which cannot be regenerated. We therefore pin the released subset by identifier: **264 validation sequences (137 harmful / 127 benign) and 528 test sequences (263 harmful / 265 benign)**, balanced across the three task types. A frozen list of 10 problem rows (for example, a pair with identical steps and opposite labels) is kept in the primary analysis for faithful replication and dropped in a sensitivity analysis."));
add(...table(["Split", "QA (h / b)", "agent (h / b)", "image (h / b)", "Total"],
  [["Validation", "43 / 43", "42 / 43", "52 / 41", "264 (137 / 127)"],
   ["Test", "86 / 90", "88 / 88", "89 / 87", "528 (263 / 265)"]],
  [2000, 1840, 1840, 1840, 1840], "**Table 1.** Evaluation subset (sequences; h = harmful, b = benign)."));
add(P("Sequences are short. In the test set the median length is 6 steps for QA, 7 for agent and 10 for image; the median harm index is 3 (QA), 6 (agent) and 10 (image)."));

add(H2("3.2 Monitor and scoring protocol"));
add(P("The judge is meta-llama/Llama-3.1-8B-Instruct (revision 0e9e39f), run locally with Hugging Face Transformers in bfloat16 on one NVIDIA H100 NVL, with greedy decoding and a 2,000-token output limit. Prompt formatting follows the upstream code: earlier steps are joined by newlines, apostrophes are removed, and the system prompt is \"You are a helpful assistant.\" Step 1 is never scored. A sequence fires at its first scored step whose score passes the threshold. We report:"));
add(B("**Timely recall:** harmful sequences first flagged at or before the harm step, divided by harmful sequences. This equals the paper's detection success rate (DSR)."));
add(B("**Sequence false-positive rate (FPR):** benign sequences flagged at any step, divided by benign sequences."));
add(B("**F1** under the paper's protocol, where a true positive is a timely detection."));
add(P("The monitor never sees the original goal, the label, the harm index or any future step. A guard in the runner raises an error if any selected history item comes from the current or a later step."));

add(H2("3.3 Phase 2: reproduction"));
add(P("We ran three upstream prompts: a zero-shot binary prompt (BINARY_BASELINE), a zero-shot probability prompt (BINARY_INTENTION_EVAL_ORIGINAL) and the paper's optimized prompt with eight in-context agent examples (BINARY_INTENTION_EVAL_W_8_AGENT_ICL, \"Agent-ICL\"). Each was run once on validation and once on test. The threshold was chosen on validation as the F1-optimal point on a 0.05 to 0.95 grid. The upstream threshold sweep contains a one-step offset; we emulate it as the primary selector and report an aligned sweep alongside it. For comparison with the authors' released outputs, which pick the threshold on the test split, we also re-score our per-sequence scores with their exact rule."));

add(H2("3.4 Phase 3: controlled history selection"));
add(P("**Fixed judge.** Agent-ICL in the paper's format (the best prompt on validation in Phase 2). The only input that changes between conditions is which earlier steps appear in the prompt."));
add(P("**Conditions.** Ten pilot conditions were specified in the analysis plan before any test run:"));
add(B("**current:** no earlier steps."));
add(B("**full:** all earlier steps."));
add(B("**recent_k:** the k most recent earlier steps."));
add(B("**bm25_k:** top k by BM25 (k1 = 1.2, b = 0.75) between the current request and each earlier step."));
add(B("**dense_k:** top k by cosine similarity under all-MiniLM-L6-v2 (revision 1110a24), no fine-tuning; each text is embedded once per run."));
add(B("**hybrid_k:** reciprocal-rank fusion (constant 60) of the relevance-sorted BM25 and dense rankings."));
add(P("Each selector was run with k = 3 and k = 5. Ties go to the more recent step, and the selected steps are presented in chronological order. Retrieval is restricted to earlier steps of the same sequence."));
add(P("**Readouts.** After an amendment (Section 3.5), two readouts are reported from the same greedy generation:"));
add(B("**Readout A (primary):** the judge's probability of answering \"1\" rather than \"0\" at its first generated token, p = P(\"1\") / (P(\"0\") + P(\"1\")), using both plain and space-prefixed digit tokens. Thresholds are restricted to [0, 1] and fitted per condition on validation as the lowest threshold whose sequence FPR is at most 0.05."));
add(B("**Readout B (secondary):** the parsed reply at a fixed cutoff of 1 (the judge's own \"harmful\" verdict), with no calibration."));
add(P("**Hypotheses** (specified in the plan; primary metric = test timely recall at the frozen validation threshold):"));
add(B("**H1:** full history beats current-only."));
add(B("**H2:** top-k retrieval keeps timely recall within a margin of full history (non-inferiority: the lower 95% bound of candidate minus full exceeds −margin)."));
add(B("**H4:** BM25 and dense retrieval beat recency at the same k."));
add(P("The margin and primary k were left open in the plan, to be set from validation results only before any test run. They were fixed at **margin 0.05** and **k = 3** after the validation run and before the test run (Section 4.3). Hypothesis H3, about oracle evidence, belongs to the mechanism phase and is not tested here."));
add(P("**Statistics.** Paired sequence bootstrap on identical test identifiers (2,000 draws, seed 1729), resampling harmful and benign sequences separately, with 95% percentile intervals. The resampling unit is the sequence, not the task family. Counts and denominators are reported throughout."));

add(H2("3.5 Analysis plan, amendment and freeze"));
add(P("The plan was written and hashed before any Phase 3 run. A first validation run under the original protocol, which calibrated thresholds on the parsed scores, failed in an informative way. The judge's parsed scores are almost binary, so with 127 benign validation sequences a 5% budget allows at most 6 flags. Conditions that flagged 7 or more benign sequences at the judge's \"1\" verdict were pushed to a threshold above every real score (in one case, a misparsed value of 3.0), so their timely recall fell to zero."));
add(P("That run was superseded and is never cited as a result. A written amendment, made before any test data were seen and referencing the plan's hash, replaced the primary readout with the continuous first-token probability (readout A), kept the parsed verdict as a fixed-cutoff secondary readout (readout B), and restricted thresholds to [0, 1]. The freeze file records the margin, k, each condition's threshold and the hash of every validation file. The freeze script refuses to overwrite it and rejects files from the superseded protocol."));
add(...table(["Time (UTC, 3 Nov 2026)", "Event"],
  [["03:43", "Analysis plan written and hashed (conditions, metric, hypotheses, margin rule)"],
   ["≈04:23*", "Validation run 798493: calibration cliff; superseded, never frozen"],
   ["04:27", "Amendment 1: readouts A and B; thresholds restricted to [0, 1]; no test data seen"],
   ["≈18:42*", "Validation run 798524 under the amendment"],
   ["18:44", "Thresholds frozen; primary k = 3; margin 0.05"],
   ["≈20:02*", "Test run 798699, executed once"]],
  [2400, 6960], "**Table 2.** Phase 3 timeline. *Run times are the timestamps of the saved result files; the other times are recorded in the plan, amendment and freeze files."));

// 4 Results
add(H1("4. Results"));
add(H2("4.1 Reproduction of the published monitor"));
add(...figure("fig1_reproduction.png", 1280, 600, "**Figure 1.** Test F1 under the paper's protocol with a validation-fixed threshold. Ours: point estimate and 95% sequence-bootstrap interval. Paper: published value ± 1 SE. The paper reports no Llama-3.1-8B number for the probability prompt."));
add(...table(["Prompt", "TP / FP / FN / TN", "F1 [95% CI]", "DSR [95% CI]", "Benign flagged", "Paper"],
  [["Binary", "161 / 3 / 102 / 262", "0.754 [0.709, 0.798]", "0.612 [0.555, 0.672]", "3 / 265", "0.719 ± 0.026 (T2); 0.690 ± 0.028 (T4)"],
   ["Probability", "225 / 42 / 38 / 223", "0.849 [0.815, 0.881]", "0.856 [0.812, 0.896]", "42 / 265", "none"],
   ["Agent-ICL", "228 / 17 / 35 / 248", "0.898 [0.869, 0.924]", "0.867 [0.826, 0.907]", "17 / 265", "0.881 ± 0.014 (T4)"]],
  [1300, 1700, 1600, 1600, 1100, 2060], "**Table 3.** Phase 2 test results (528 sequences, one run per prompt, threshold from validation)."));
add(...table(["Prompt", "Ours, their rule (TP / FP / FN / TN)", "Ours F1", "Authors' saved file", "Their F1"],
  [["Probability (threshold 0.25)", "232 / 35 / 31 / 230", "0.875", "242 / 49 / 21 / 216", "0.874"],
   ["Agent-ICL (threshold 0.05)", "237 / 9 / 26 / 256", "0.931", "245 / 18 / 18 / 247", "0.932"]],
  [2400, 2200, 1000, 2200, 1560], "**Table 4.** Like-for-like check against the authors' released test outputs, both scored with the upstream sweep rule. That rule picks the threshold on test and is optimistic; these are not our official results."));
add(P("The probability and Agent-ICL prompts reproduce: F1 matches the authors' released outputs to within 0.001 under the same rule, and our Agent-ICL test F1 is within about one standard error of the published value. Our local model flags slightly fewer sequences than the authors' API-served model (fewer true and false positives), so the match is in F1 rather than per-sequence behavior. The binary prompt is consistent with the paper's Table 2 (about 1 SE) and borderline against its Table 4 (about 1.8 SE); the paper's two values for this prompt themselves differ. Dropping the 10 problem rows changes test F1 by at most 0.003."));
add(...table(["Type", "TP / FP / FN / TN", "F1", "DSR", "Too late", "Never flagged"],
  [["QA", "56 / 0 / 30 / 90", "0.789", "0.651", "15", "15"],
   ["agent", "83 / 11 / 5 / 77", "0.912", "0.943", "0", "5"],
   ["image", "89 / 6 / 0 / 81", "0.967", "1.000", "0", "0"]],
  [1400, 2200, 1200, 1200, 1500, 1860], "**Table 5.** Agent-ICL by task type (test). QA is the hard type: every late detection is a QA sequence."));
add(P("Agent-ICL beats the probability prompt by +0.049 F1 (paired 95% CI +0.019 to +0.079). The gain comes from fewer false alarms (benign sequences flagged by only one prompt: 35 probability vs 10 Agent-ICL), not from more detections (DSR +0.011, CI −0.026 to +0.049)."));

add(H2("4.2 Readout validity"));
add(...table(["Check", "Validation (per condition)", "Test (per condition)"],
  [["Replies parsed", "1,744 / 1,744", "all"],
   ["First-token P(0) + P(1): minimum", "≥ 0.992", "median ≈ 1.000"],
   ["Decisions with P(0) + P(1) < 0.5", "0", "0"],
   ["A (p ≥ 0.5) agrees with parsed 0/1 verdict", "99.7% to 99.9%", "99.5% to 99.8%"]],
  [3800, 2780, 2780], "**Table 6.** The continuous readout is a faithful version of the judge's own verdict."));
add(P("Unlike the parsed scores, readout A spreads out among benign sequences (for the full condition on validation, 109 of 127 benign sequences have a maximum score below 0.1 and 18 lie between 0.1 and 0.99), which is what fixed-FPR calibration needs. Every condition reached exactly 6 of 127 benign flags (FPR 0.047) on validation."));

add(H2("4.3 Validation stage and the choice of k and margin"));
add(...table(["Condition", "Threshold", "FPR", "Timely recall", "Paired diff vs full [95% CI]"],
  [["current", "0.989", "0.047", "0.693", "−0.153 [−0.226, −0.088]"],
   ["full", "0.438", "0.047", "0.847", "reference"],
   ["recent_k3", "0.651", "0.047", "0.839", "−0.007 [−0.036, +0.022]"],
   ["bm25_k3", "0.562", "0.047", "0.839", "−0.007 [−0.036, +0.015]"],
   ["dense_k3", "0.469", "0.047", "0.847", "+0.000 [−0.029, +0.029]"],
   ["hybrid_k3", "0.500", "0.047", "0.854", "+0.007 [−0.022, +0.036]"],
   ["recent_k5", "0.469", "0.047", "0.832", "−0.015 [−0.036, +0.000]"],
   ["bm25 / dense / hybrid k5", "0.438", "0.047", "0.847", "+0.000 (0 discordant)"]],
  [2400, 1300, 1100, 1500, 3060], "**Table 7.** Validation, readout A (127 benign / 137 harmful). Used only for calibration and for the k and margin decisions."));
add(P("Two observations set the remaining choices. First, at k = 5 about 70% of decisions already see every earlier step, and BM25, dense and hybrid at k = 5 matched full history on every validation sequence; k = 5 therefore cannot separate retrieval from full history, so **k = 3** was confirmed as primary (at k = 3, 55% of decisions see less than full history). Second, the margin of **0.05** was set, after this validation run and before any test run, as the largest loss of timely recall we would accept in practice. With about 263 harmful test sequences, the expected paired standard error is about 0.011, so a true difference near zero passes comfortably and a true loss of 5 points or more almost never passes. The test is not sensitive to smaller losses: a true loss of 3 points would still pass about 44% of the time."));

add(H2("4.4 Planned test results"));
add(...table(["Condition", "Threshold", "Benign flagged (FPR)", "Timely recall", "Any detection"],
  [["current", "0.989", "5 (0.019)", "0.684", "0.757"],
   ["full", "0.438", "18 (0.068)", "0.798", "0.852"],
   ["recent_k3", "0.651", "12 (0.045)", "0.772", "0.856"],
   ["bm25_k3", "0.562", "15 (0.057)", "0.802", "0.871"],
   ["dense_k3", "0.469", "17 (0.064)", "0.810", "0.875"],
   ["hybrid_k3", "0.500", "16 (0.060)", "0.814", "0.882"],
   ["recent_k5", "0.469", "16 (0.060)", "0.791", "0.856"],
   ["bm25_k5", "0.438", "18 (0.068)", "0.802", "0.859"],
   ["dense_k5", "0.438", "18 (0.068)", "0.795", "0.852"],
   ["hybrid_k5", "0.438", "18 (0.068)", "0.798", "0.856"]],
  [2200, 1400, 2200, 1700, 1860], "**Table 8.** Test results, readout A, frozen thresholds (263 harmful / 265 benign). One run."));
add(...figure("fig2_operating_points.png", 1280, 800, "**Figure 2.** Test operating points under the frozen thresholds. The dashed line is the validation FPR budget; points to its right exceeded the budget on test. Grey points are the k = 5 selectors."));
add(...figure("fig3_forest_vs_full.png", 1280, 760, "**Figure 3.** Paired difference in timely recall against full history (test, readout A), with 95% paired-bootstrap intervals. Green: the H2 candidates. Dashed: the non-inferiority margin."));

add(P("Phase 3's full condition (timely recall 0.798 with 18 benign flags) detects less than Phase 2's Agent-ICL run (0.867 with 17), although the prompt and history are the same. The two use different readouts and threshold rules: Phase 2 thresholds the judge's parsed reply at the F1-optimal validation point, while Phase 3 thresholds the first-token probability at a 5% validation false-alarm budget. We did not isolate the cause. Phase 3's readout B, the parsed verdict at a fixed cutoff, also gives a lower full-history recall (0.776 at 14 benign flags, Table 11), so the readout alone may not explain the gap. Phase 3's full condition should be read as a different scoring of the same prompt, not as a rerun of the Phase 2 monitor."));
add(H3("H2: top-3 retrieval vs full history (primary)"));
add(...table(["Candidate", "Timely recall diff [95% CI] (SE)", "Lower bound > −0.05?", "FPR diff [95% CI]", "Discordant (full only / cand. only)"],
  [["bm25_k3", "+0.004 [−0.011, +0.019] (0.008)", "Yes", "−0.011 [−0.034, +0.011]", "2 / 3"],
   ["dense_k3", "+0.011 [−0.008, +0.034] (0.010)", "Yes", "−0.004 [−0.015, +0.008]", "2 / 5"],
   ["hybrid_k3", "+0.015 [−0.004, +0.038] (0.011)", "Yes", "−0.008 [−0.023, +0.008]", "2 / 6"]],
  [1500, 2600, 1500, 2160, 1600], "**Table 9.** H2 non-inferiority tests (candidate minus full)."));
add(P("**H2 is supported for all three retrievers.** Each is non-inferior to full history in timely recall, and none flags more benign sequences. The conclusion survives a Bonferroni split across the three candidates, since the worst lower bound (−0.011) is far from −0.05; each arm would also pass any margin larger than 0.011 (BM25), 0.008 (dense) or 0.004 (hybrid) on the unadjusted 95% bounds, or about 0.015, 0.013 and 0.011 on Bonferroni-adjusted bounds. The result is specific to this benchmark's short sequences: among the decisions that matter for timely detection (steps up to the harm step), top-3 input differed from full history in only 14% of harmful QA decisions, 45% of harmful agent decisions and 67% of harmful image decisions (post-hoc count; Table 14)."));
add(H3("H1: full history vs current request only"));
add(P("Full history detects more attacks in time: **+0.114 (95% CI +0.068 to +0.156)**, with 34 sequences caught in time only with full history and 4 only without it. It also flags more benign sequences: **+0.049 FPR (95% CI +0.019 to +0.079)**, 18 vs 5. Both thresholds were fitted to the same validation FPR, but they transferred to different test FPRs, so the planned comparison is not made at equal false alarms. We apply one rule to H1 and H4: a timely-recall gain on readout A that comes with a higher test FPR is reported as positive on the primary readout but not separable from the false-alarm difference, and we report whether it also holds under readout B. **H1 is positive on the primary readout but not separable from the false-alarm difference.** It does not hold under readout B, where the current-only monitor detects more than full history at an FPR of 0.313 (Table 11). Section 4.6 shows, exploratorily, that the gain disappears at equal false alarms."));
add(H3("H4: retrieval vs recency at k = 3"));
add(...table(["Candidate − recent_k3", "Readout A: timely diff", "Readout A: FPR diff", "Readout B: timely diff", "Readout B: FPR diff"],
  [["bm25_k3", "+0.030 [+0.008, +0.057]", "+0.011 [0.000, +0.026]", "−0.008 [−0.023, +0.008]", "+0.004 [0.000, +0.011]"],
   ["dense_k3", "+0.038 [+0.015, +0.065]", "+0.019 [+0.004, +0.038]", "−0.004 [−0.019, +0.008]", "−0.004 [−0.011, 0.000]"]],
  [1800, 1890, 1890, 1890, 1890], "**Table 10.** H4 comparisons (paired, 95% CI)."));
add(P("On readout A, BM25 and dense retrieval detect 3 to 4 points more than recency, but they also flag more benign sequences, because recency's frozen threshold (0.651) transferred to a lower test FPR (0.045). Under one fixed cutoff for every condition (readout B), the difference disappears. By the same rule as H1, **H4 is positive on readout A but not separable from the false-alarm difference** (for BM25 the FPR interval touches zero), and like H1 it does not hold under readout B, so we do not count it as support for retrieval over recency."));
add(P("Retrieval and recency often, but not always, saw different steps (post-hoc, Table 14). Whenever a decision had three or fewer earlier steps, every k = 3 selector saw the full history, so the two inputs were identical; in the timely window this covers 86% of QA, 55% of agent and 33% of image harmful decisions (47% overall). Over decisions with more than three earlier steps, a retriever's top 3 equalled the three most recent steps in only 6% to 29% of decisions, with mean Jaccard overlap 0.34 to 0.57. Of the 12 to 15 harmful sequences on which a retriever and recency disagreed about timely detection, 6 to 9 involved different selected steps within the relevant window; the other 5 or 6 saw identical inputs throughout, so their disagreement came from the different thresholds alone."));

add(H2("4.5 Secondary results"));
add(...table(["Condition", "FPR", "Timely recall", "Diff vs full [95% CI]"],
  [["current", "0.313", "0.913", "+0.137 [+0.095, +0.183]"],
   ["full", "0.053", "0.776", "reference"],
   ["recent_k3", "0.057", "0.810", "+0.034 [+0.015, +0.057]"],
   ["bm25_k3", "0.060", "0.802", "+0.027 [+0.008, +0.049]"],
   ["dense_k3", "0.053", "0.806", "+0.030 [+0.008, +0.053]"],
   ["hybrid_k3", "0.053", "0.814", "+0.038 [+0.015, +0.065]"],
   ["k = 5 selectors", "0.049", "0.779 to 0.783", "+0.004 to +0.008"]],
  [2400, 1600, 2000, 3360], "**Table 11.** Readout B: the judge's own verdict at a fixed cutoff, no calibration (test)."));
add(P("Without calibration, the current-only monitor flags almost a third of benign sequences (83 of 265). At its own verdict, k = 3 selection catches 2.7 to 3.8 points more attacks than full history at a similar FPR, so at the judge's own verdict, less history did not cost detections. Because readout B is uncalibrated, this is reported as secondary."));
add(P("**Calibration transfer.** A validation fit of 6 of 127 benign flags gave test FPRs from 0.019 (current) to 0.068 (full and the BM25, dense and hybrid k = 5 conditions). Full history exceeded the 0.05 budget on test. With 127 benign calibration sequences, one sequence moves the FPR by 0.8 points, so the budget is estimated coarsely."));
add(P("**Cost.** Mean judge input tokens per decision were 1,477 (current), about 1,525 (k = 3), about 1,544 (k = 5) and 1,556 (full). The fixed few-shot prompt dominates the input, so k = 3 retrieval saves only about 2% of input tokens here (Figure 6), and with prefix caching, which reuses the shared prompt and earlier history across steps, full history would be cheaper still relative to retrieval, whose selected context changes from step to step. Dense retrieval adds one embedding per text. Wall-clock seconds per decision depended on the cluster node and are not compared."));
add(...figure("fig6_recall_vs_tokens.png", 1280, 680, "**Figure 6.** Timely recall against mean judge input tokens per decision (test). The x-axis spans only about 80 tokens."));

add(H2("4.6 Exploratory analyses (decided after unblinding)"));
add(P("The analyses in this section were chosen after the test results were seen and tune thresholds on test. They are not confirmatory."));
add(...figure("fig4_matched_false_alarms.png", 1280, 760, "**Figure 4.** EXPLORATORY. Timely recall as a function of the number of benign test sequences flagged, with the threshold tuned on test; dots mark the thresholds frozen before the test run."));
add(...table(["Condition", "≤ 5 benign flags", "≤ 13 benign flags", "≤ 18 benign flags"],
  [["current", "0.684 (5)", "0.772 (12)", "0.787 (18)"],
   ["full", "0.658 (5)", "0.772 (13)", "0.798 (18)"],
   ["recent_k3", "0.711 (5)", "0.787 (13)", "0.810 (18)"],
   ["bm25_k3", "0.700 (5)", "0.760 (13)", "0.802 (18)"],
   ["dense_k3", "0.700 (5)", "0.802 (13)", "0.810 (17)"],
   ["hybrid_k3", "0.707 (5)", "0.810 (13)", "0.821 (17)"]],
  [2400, 2320, 2320, 2320], "**Table 12.** EXPLORATORY. Timely recall at matched benign-flag counts (actual count in parentheses)."));
add(P("At equal false alarms, **full history does not beat current-only overall**, recency is not worse than retrieval, and k = 3 is within about one point of full. The H1 gain in Section 4.4 therefore reflects a different operating point as much as better discrimination overall."));
add(...figure("fig5_by_task_type.png", 1320, 640, "**Figure 5.** Timely recall (left) and benign sequence FPR (right) by task type at the frozen thresholds (test, readout A)."));
add(...table(["Comparison", "Type (harmful / benign)", "Timely diff [95% CI]", "FPR diff [95% CI]", "Discordant"],
  [["full − current", "QA (86 / 90)", "+0.198 [+0.093, +0.302]", "0.000", "3 / 20"],
   ["full − current", "agent (88 / 88)", "+0.148 [+0.080, +0.227]", "+0.057 [0.000, +0.114]", "0 / 13"],
   ["full − current", "image (89 / 87)", "0.000 [−0.034, +0.034]", "+0.092 [+0.023, +0.161]", "1 / 1"],
   ["hybrid_k3 − recent_k3", "QA", "+0.093 [+0.035, +0.163]", "0.000", "0 / 8"],
   ["hybrid_k3 − recent_k3", "agent", "+0.034 [−0.023, +0.091]", "+0.011 [0.000, +0.034]", "2 / 5"],
   ["hybrid_k3 − full", "QA", "0.000 [−0.035, +0.035]", "0.000", "1 / 1"]],
  [2000, 1700, 2000, 2000, 1660], "**Table 13.** EXPLORATORY. Paired differences by task type at the frozen thresholds."));
add(P("The per-type view resolves the apparent conflict between Sections 4.4 and 4.6."));
add(B("**Decomposed QA** is where history helps cleanly: no condition flags any benign QA sequence, and full history raises timely QA recall from 0.291 to 0.488. Hybrid retrieval matches full history here and beats recency by 9 points (8 sequences to 0). Their inputs were identical in most QA decisions (Table 14), so this difference may come largely from their different thresholds."));
add(B("**Agent** detection also improves with history (+0.148), but benign agent FPR rises from 0.034 to 0.091."));
add(B("**Image** attacks are caught almost perfectly with or without history, while benign image FPR rises from 0.023 to 0.115 with full history."));
add(P("Overall, history buys QA and agent detections at the price of benign agent and image false alarms. When the threshold is moved to equalize false alarms, the two effects roughly cancel."));
add(P("**Post-hoc input checks (added in revision).** Table 14 recomputes which earlier steps each selector chose, using the same deterministic selectors on the same records, and reads the judge's scores from the saved test files. No new judge calls were made (script scripts/phase3_overlap.py)."));
add(...table(["Check", "QA", "agent", "image"],
  [["Harmful decisions up to the harm step where top-3 input differs from full", "29 / 203 (14%)", "217 / 480 (45%)", "534 / 801 (67%)"],
   ["Benign decisions where top-3 input differs from full", "148 / 401 (37%)", "229 / 490 (47%)", "518 / 779 (66%)"],
   ["Top 3 identical to 3 most recent, BM25 / dense / hybrid (decisions with > 3 earlier steps)", "13% / 12% / 12% (n = 365)", "20% / 29% / 27% (n = 479)", "6% / 9% / 9% (n = 1,052)"],
   ["Mean Jaccard overlap with 3 most recent, BM25 / dense / hybrid", "0.46 / 0.45 / 0.45", "0.52 / 0.57 / 0.57", "0.34 / 0.40 / 0.39"],
   ["Harmful sequences where retriever and recent_k3 disagree on timely detection: different steps / same steps (BM25, dense, hybrid; all types)", "7 / 5, 6 / 6, 9 / 6", "", ""]],
  [4200, 1720, 1720, 1720], "**Table 14.** POST-HOC. Selected inputs at k = 3. Identical-input noise proxy: across 1,562 to 1,833 decisions per pair of conditions (full, recent_k3 and each k = 3 retriever) where the recomputed inputs were identical in separate condition passes of the single test job, no saved score differed by more than 0.001 (the largest difference rounds to 0.0000). This agreement is consistent with the recomputed selections matching the inputs the judge actually received."));

// 5 Discussion
add(H1("5. Discussion"));
add(P("**A small history is enough here, on short sequences.** The confirmatory result is H2: giving the judge three selected earlier steps matches full history in timely detection, with no increase in false alarms detected, for every retriever tested, on the planned primary readout and on the fixed-cutoff readout. The result is partly built in, because top-3 input often equals full history on this benchmark, especially for the QA attacks where history matters most. For a monitor that must limit what it reads, it is an encouraging direction, not yet evidence for long histories."));
add(P("**History is not free.** At the frozen thresholds, full history recovers decomposed QA attacks that the current request alone hides, and it raised false alarms on benign agent and image sequences (exploratory, Table 13). Because the two conditions use different thresholds (0.438 vs 0.989), this does not show that the judge itself becomes more suspicious; the operating points alone can produce it. A deployment that adds history to a monitor should expect to recalibrate, and should measure benign false alarms per task type rather than only overall. Per-type timeliness claims are also limited by the labels: the image harm step is always the last step and agent harm steps were annotated by a language model, so only QA tests timeliness in a demanding way."));
add(P("**Retrieval vs recency is unresolved on this benchmark.** Retrieval and recency often chose different steps once more than three earlier steps were available, but their inputs were identical in roughly half of the decisions that decide timely detection, and neither detected more attacks once false alarms were held fixed. On short sequences top-3 input often equals full history, so the benchmark cannot show retrieval's advantage on long histories, or its cost advantage. The QA-only hint that hybrid retrieval beats recency is a hypothesis for a follow-up with a protocol fixed in advance on longer traces, not a finding."));
add(P("**Calibration on small benign sets is fragile.** Two problems appeared, both about calibration rather than detection: near-binary scores made fixed-FPR calibration impossible until the continuous readout was used, and thresholds fitted on 127 benign sequences transferred to test FPRs between 1.9% and 6.8%. Comparisons between monitors at a nominal FPR are only as good as the benign calibration set."));

// 6 Limitations
add(H1("6. Limitations"));
add(B("**Test reuse.** The 528 test sequences were also evaluated in Phase 2. No Phase 3 choice used them, the reuse was disclosed in the plan before any Phase 3 run, and a fresh hold-out would have been 172 unbalanced rows."));
add(B("**Threshold transfer** confounds H1 and H4 on readout A. A larger benign calibration set, or a comparison at a fixed test FPR specified in advance, would remove it."));
add(B("**Short sequences** limit what history selection can show, and make cost savings negligible."));
add(B("**No external registration.** The analysis plan, its amendment and the frozen thresholds were fixed before the test run and are linked by SHA-256 hashes, but they were not deposited with an external registry, so their timing rests on the project's own records."));
add(B("**One judge, one run.** One 8B model with greedy decoding. A full repeat run was not made, since that would reuse the test set. As a proxy, wherever two conditions sent the judge identical inputs in their separate passes of the single test job, no saved score differed by more than 0.001 (Table 14), so decoding noise for identical inputs is unlikely to explain the differences reported here."));
add(B("**Padding.** An attacker could insert harmless steps to push the relevant evidence out of a top-k window; selected-history monitors were not tested against this."));
add(B("**Statistics** use a sequence-level bootstrap. Benign sequences were derived from harmful ones and some goals repeat, so task-family clustering would likely widen the intervals."));
add(B("**Labels** are taken as given. The harm index is built differently per task type, and for agent tasks it was annotated by a language model."));
add(B("**Protocol amendment.** The primary readout changed after a first validation run, never on test. The reasons, the evidence and the superseded files are recorded."));
add(B("**Mechanism untested.** This study shows that decisions change with history, not why; separating retrieval failures from judge misinterpretation needs independent evidence annotations (H3)."));

// 7 Conclusion
add(H1("7. Conclusion and next experiment"));
add(P("On DecomposedHarm, with a reproduced Llama-3.1-8B sequential monitor, top-3 retrieval of earlier requests is non-inferior to full history in timely detection, with no increase in false alarms detected, on sequences short enough that top-3 often equals full history. In exploratory analyses, history helps most on QA and agent attacks and comes with more false alarms on benign agent and image tasks, so at equal false alarms its overall benefit disappears. Retrieval did not beat recency here; the two often chose different steps on longer prefixes, but saw identical inputs in roughly half of the decisions that matter."));
add(P("The next planned experiment is a mechanism study. On a small, pre-specified subset with independent human annotations of which earlier steps carry the attack, it would give the judge exactly that evidence (an oracle) and remove, restore or replace individual steps with matched benign controls. Its aim is to separate two failure modes: the right evidence not being retrieved, and the judge misreading evidence it was given, including the benign false alarms that appear with full history at the frozen thresholds."));

// Availability, ethics, references
add(H1("Data, code and ethics"));
add(P("**Code.** The full experiment code, tests (104 passing), configuration, analysis plan, amendment, frozen thresholds and analysis scripts are public at github.com/SatyaChamana/stateful-safety-monitor-study, release phase3-v1.1 (the original study state is tag phase3-v1 of the private working repository). Every number in this manuscript, including the post-hoc Table 14 (scripts/phase3_overlap.py), is produced by scripts in that repository from local result files."));
add(P("**Data.** DecomposedHarm is gated and evaluation-only. Raw benchmark text, per-sequence result files and model weights are kept out of version control, and no benchmark text appears in this manuscript. Configuration files contain row identifiers only."));
add(P("**Ethics.** No harmful objective was generated or executed; the monitor only classifies pre-existing benchmark requests. All inference was local; no benchmark prompt was sent to a remote service. The benchmark was never used for training."));

add(H1("References"));
const refs = [
  "Cormack, G. V., Clarke, C. L. A., and Buettcher, S. (2009). Reciprocal rank fusion outperforms Condorcet and individual rank learning methods. SIGIR 2009.",
  "Glukhov, D., et al. (2024). Breach by a thousand leaks: Unsafe information leakage in \"safe\" AI responses. arXiv:2407.02551.",
  "Grattafiori, A., et al. (2024). The Llama 3 herd of models. arXiv:2407.21783.",
  "Greenblatt, R., Shlegeris, B., Sachan, K., and Roger, F. (2023). AI control: Improving safety despite intentional subversion. arXiv:2312.06942.",
  "Jones, E., Dragan, A., and Steinhardt, J. (2024). Adversaries can misuse combinations of safe models. arXiv:2406.14595.",
  "Reimers, N., and Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. EMNLP 2019.",
  "Robertson, S., and Zaragoza, H. (2009). The probabilistic relevance framework: BM25 and beyond. Foundations and Trends in Information Retrieval, 3(4).",
  "Russinovich, M., Salem, A., and Eldan, R. (2024). Great, now write an article about that: The Crescendo multi-turn LLM jailbreak attack. arXiv:2404.01833.",
  "Souly, A., et al. (2024). A StrongREJECT for empty jailbreaks. arXiv:2402.10260.",
  "Wang, W., et al. (2020). MiniLM: Deep self-attention distillation for task-agnostic compression of pre-trained transformers. NeurIPS 2020.",
  "Chen, Y.-H., Joshi, N., Chen, Y., Andriushchenko, M., Angell, R., & He, H. (2025). Monitoring decomposition attacks in LLMs with lightweight sequential monitors. arXiv:2506.10949 (ICLR 2026).",
];
refs.forEach((r) => add(new Paragraph({ children: runs(r, { size: 21 }), spacing: { after: 80 },
  indent: { left: 360, hanging: 360 } })));

// Appendices
add(new Paragraph({ children: [new PageBreak()] }));
add(H1("Appendix A. Setup and deviations from the paper"));
add(...table(["Item", "Value"],
  [["Dataset", "YuehHanChen/DecomposedHarm @ 25e1d5910c2cccbdcae11799193595cea999bda8"],
   ["Upstream code", "YuehHanChen/Monitoring-Decomposition-Attack @ 79f5b4109aec8ff88f92ffe2d7a94eb419699d68"],
   ["Judge", "meta-llama/Llama-3.1-8B-Instruct @ 0e9e39f249a16976918f6564b8830bc894c89659, bfloat16"],
   ["Encoder", "sentence-transformers/all-MiniLM-L6-v2 @ 1110a243fdf4706b3f48f1d95db1a4f5529b4d41"],
   ["Hardware", "NVIDIA H100 NVL (95,830 MiB), driver 610.43.02, PyTorch 2.15.0+cu132"],
   ["Decoding", "greedy, max_new_tokens 2,000"],
   ["Jobs", "Phase 2: 798355, 798359, 798360; Phase 3: validation 798524, test 798699; superseded 798493"]],
  [2000, 7360], "**Table A1.** Pinned setup."));
add(N("**Serving:** local Transformers in bfloat16 vs the paper's API-served model (quantization not verified)."));
add(N("**Chat template:** the local tokenizer's Llama 3.1 template; the API's server-side formatting is not observable."));
add(N("**Nondeterminism:** greedy GPU decoding is not bit-identical across runs."));
add(N("**Threshold search:** the upstream sweep has a one-step offset; we emulate it as primary and report an aligned sweep beside it."));
add(N("**Problem rows** kept in the primary analysis (sensitivity analysis drops them)."));
add(N("**Parsing:** upstream rule (last number in the reply). 11 of 5,202 Phase 2 replies had no number and were mapped to benign, as upstream does."));

add(H1("Appendix B. Reproducing the numbers"));
["python scripts/summarize_paper_runs.py results/paper_eval_*.json",
 "python scripts/summarize_phase3.py results/phase3a1_test_*.json",
 "python scripts/summarize_phase3.py --reference recent_k3 results/phase3a1_test_*.json",
 "python scripts/phase3_exploratory.py results/phase3a1_test_*.json",
 "PYTHONPATH=src python scripts/make_figures.py",
 "PYTHONPATH=src python scripts/phase3_overlap.py > docs/phase3_overlap.md   # Table 14 (post-hoc)"].forEach((c) =>
  add(new Paragraph({ children: [new TextRun({ text: c, font: "Courier New", size: 18 })], spacing: { after: 60 } })));
add(P("Inputs: configs/phase3_plan.json, configs/phase3_amendment_1.json, configs/phase3_thresholds.json (which records the SHA-256 of every validation file). The full change history is in docs/experiment_log.md."));

// ---------- document ----------
const doc = new Document({
  creator: "Satya Chamana",
  title: "How Much History Does a Sequential Misuse Monitor Need?",
  styles: {
    default: { document: { run: { font: FONT, size: 22 }, paragraph: { spacing: { line: 276 } } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 28, bold: true }, paragraph: { spacing: { before: 320, after: 140 }, keepNext: true, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 24, bold: true }, paragraph: { spacing: { before: 240, after: 100 }, keepNext: true, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 22, bold: true, italics: true }, paragraph: { spacing: { before: 180, after: 80 }, keepNext: true, outlineLevel: 2 } },
    ],
  },
  numbering: { config: [
    { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 540, hanging: 270 } } } },
      { level: 1, format: LevelFormat.BULLET, text: "◦", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 1080, hanging: 270 } } } }] },
    { reference: "numbers", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 540, hanging: 300 } } } }] },
  ] },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
    headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT,
      children: [new TextRun({ text: "History selection for sequential misuse monitors (draft, phase3-v1.1)", size: 16, italics: true, color: "666666" })] })] }) },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ children: [PageNumber.CURRENT], size: 18 })] })] }) },
    children: body,
  }],
});
Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(OUT, buf); console.log("wrote", OUT); });
