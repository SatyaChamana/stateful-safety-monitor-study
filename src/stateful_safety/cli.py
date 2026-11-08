"""Safe CLI for dataset audit and smoke tests; no automatic remote data access."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .dataset import SCHEMAS,audit,audit_files,load_records
from .judge import MockJudge, TransformersJudge
from .metrics import summarize,calibrate_threshold
from .statistics import bootstrap_sequence_ci
from .integrity import compare_splits
from .runner import run_sequences


def _write_json(payload,path):
    if path:
        target=Path(path)
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(json.dumps(payload,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps(payload,indent=2,allow_nan=False))


def _args():
    p=argparse.ArgumentParser(description="Safety monitoring research scaffolding")
    s=p.add_subparsers(dest="command",required=True)
    for name in ("audit","pilot","evaluate"):
        sub=s.add_parser(name)
        if name == "audit":
            sub.add_argument("--data",required=True,nargs="+",
                             help="Local .jsonl/.json/.csv file(s); never auto-downloads")
            sub.add_argument("--revision",default=None,help="Upstream dataset revision to record")
        elif name == "pilot":
            sub.add_argument("--data",required=True,help="Local .jsonl/.json/.csv; never auto-downloads")
        else:
            sub.add_argument("--validation",required=True,help="Local validation dataset")
            sub.add_argument("--test",required=True,help="Local held-out dataset")
            sub.add_argument("--fpr-budget",type=float,default=0.05)
            sub.add_argument("--protocol",choices=("diagnostic","paper"),default="diagnostic",
                             help="'paper' reproduces the upstream prompt, parsing and F1 sweep")
            sub.add_argument("--paper-prompt",default="BINARY_INTENTION_EVAL_ORIGINAL",
                             help="Upstream BINARY_* template name (paper protocol only)")
            sub.add_argument("--upstream-prompts",default=None,help="Pinned upstream prompts.py")
            sub.add_argument("--max-input-tokens",type=int,default=8192)
            sub.add_argument("--ids",default=None,
                             help="Split ID file (configs/paper_eval_split.json) to restrict both splits")
            sub.add_argument("--exclude",default=None,
                             help="Frozen exclusion file; adds a sensitivity block, primary keeps all rows")
        sub.add_argument("--index-base",type=int,choices=(0,1),default=None,
                         help="Required when harmful_index is present; verify from paper")
        sub.add_argument("--schema",choices=SCHEMAS,default="native",
                         help="Row layout; 'decomposedharm' is the upstream CSV layout")
        sub.add_argument("--out",default=None)
        if name in ("pilot","evaluate"):
            sub.add_argument("--mode",choices=("current","recent","full","bm25","dense"),default="full")
            sub.add_argument("--k",type=int,default=3)
            sub.add_argument("--backend",choices=("mock","hf"),default="mock")
            sub.add_argument("--model-id",default=None)
            sub.add_argument("--model-revision",default=None,
                             help="Full 40-hex HF commit of the model; required for --backend hf")
            sub.add_argument("--dtype",default=None,choices=("bfloat16","float16","float32"),
                             help="Model weights dtype; required for --backend hf, never 'auto'")
            sub.add_argument("--dense-model",default=None)
            sub.add_argument("--threshold",type=float,default=0.5)
            sub.add_argument("--allow-mock-on-real-data",action="store_true")
    hist=s.add_parser("history-eval",help="Phase 3: one history condition, one stage")
    hist.add_argument("--stage",required=True,choices=("validation","test"))
    hist.add_argument("--data",required=True,help="Local file for THIS stage's split")
    hist.add_argument("--ids",default=None,help="Split ID file (configs/paper_eval_split.json)")
    hist.add_argument("--mode",required=True,choices=("current","recent","full","bm25","dense","hybrid"))
    hist.add_argument("--k",type=int,default=None)
    hist.add_argument("--fpr-budget",type=float,default=0.05)
    hist.add_argument("--thresholds",default=None,help="Frozen validation thresholds (test stage only)")
    hist.add_argument("--dense-model",default=None)
    hist.add_argument("--dense-revision",default=None)
    hist.add_argument("--index-base",type=int,choices=(0,1),default=None)
    hist.add_argument("--schema",choices=SCHEMAS,default="native")
    hist.add_argument("--paper-prompt",default="BINARY_INTENTION_EVAL_W_8_AGENT_ICL")
    hist.add_argument("--upstream-prompts",default=None)
    hist.add_argument("--max-input-tokens",type=int,default=8192)
    hist.add_argument("--backend",choices=("mock","hf"),default="mock")
    hist.add_argument("--model-id",default=None)
    hist.add_argument("--model-revision",default=None)
    hist.add_argument("--dtype",default=None,choices=("bfloat16","float16","float32"))
    hist.add_argument("--out",default=None)
    smoke=s.add_parser("paper-smoke",help="Validation-only paper-mode reply check; no metrics")
    smoke.add_argument("--validation",required=True,help="Local validation dataset (no test input exists)")
    smoke.add_argument("--n",type=int,default=20,help="Sequences to score, split across labels/types")
    smoke.add_argument("--ids",default=None,help="Restrict to the validation IDs in this split file")
    smoke.add_argument("--index-base",type=int,choices=(0,1),default=None)
    smoke.add_argument("--schema",choices=SCHEMAS,default="native")
    smoke.add_argument("--paper-prompt",default="BINARY_INTENTION_EVAL_ORIGINAL")
    smoke.add_argument("--upstream-prompts",default=None)
    smoke.add_argument("--max-input-tokens",type=int,default=8192)
    smoke.add_argument("--backend",choices=("mock","hf"),default="mock")
    smoke.add_argument("--model-id",default=None)
    smoke.add_argument("--model-revision",default=None)
    smoke.add_argument("--dtype",default=None,choices=("bfloat16","float16","float32"))
    smoke.add_argument("--out",default=None)
    return p.parse_args()


def main():
    args=_args()
    if args.command=="evaluate":
        return _evaluate(args)
    if args.command=="paper-smoke":
        return _paper_smoke(args)
    if args.command=="history-eval":
        return _history_eval(args)
    if args.command=="audit":
        if args.schema=="native" and len(args.data)==1 and args.revision is None:
            _write_json(audit(load_records(args.data[0],index_base=args.index_base)),args.out)
        else:
            _write_json(audit_files(args.data,schema=args.schema,index_base=args.index_base,
                                    revision=args.revision),args.out)
        return
    recs=load_records(args.data,index_base=args.index_base,schema=args.schema)
    if args.backend=="mock" and "fixture" not in Path(args.data).name and not args.allow_mock_on_real_data:
        raise SystemExit("Refusing mock evaluation on real data; mock outputs are NOT scientific results")
    if args.backend=="hf":
        if not args.model_id:
            raise SystemExit("Provide --model-id for local HF backend")
        judge=TransformersJudge(args.model_id,revision=_model_revision(args),dtype=args.dtype)
    else:
        judge=MockJudge()
    scores,usage=run_sequences(recs,judge,args.mode,k=args.k,dense_model=args.dense_model)
    metrics=summarize(recs,scores,args.threshold)
    payload={"status":"SMOKE_TEST_ONLY" if args.backend=="mock" else "UNVALIDATED_PILOT",
             "dataset_path_redacted":Path(args.data).name,"backend":judge.name,
             "model_revision":args.model_revision,"dtype":args.dtype,
             "history_mode":args.mode,"k":args.k,"metrics":metrics,"usage":usage,
             "warnings":["This command does not reproduce the paper's published evaluator.",
                         "No score thresholds were calibrated unless done separately on validation.",
                         "Do not interpret mock-label performance as research evidence." if args.backend=="mock"
                         else "Verify model output parsing, prompting and licensed benchmark access."]}
    _write_json(payload,args.out)


def _restrict(records, ids_file, split):
    """Keep exactly the listed IDs; every listed ID must exist."""
    wanted=set(json.loads(Path(ids_file).read_text())["splits"][split]["ids"])
    kept=[r for r in records if r.sequence_id in wanted]
    if len(kept)!=len(wanted):
        raise ValueError(f"{split}: {len(wanted)-len(kept)} listed IDs missing from data")
    return kept


def _evaluate(args):
    """Separate validation threshold fitting from held-out test evaluation."""
    validation = load_records(args.validation,index_base=args.index_base,default_split="validation",
                              schema=args.schema)
    test = load_records(args.test,index_base=args.index_base,default_split="test",schema=args.schema)
    if args.ids:
        validation,test=_restrict(validation,args.ids,"validation"),_restrict(test,args.ids,"test")
    split_check = compare_splits(validation, test)
    if not split_check["valid_for_initial_evaluation"]:
        raise ValueError("Validation/test split overlap detected; review source partitions")
    if args.protocol == "paper":
        return _evaluate_paper(args, validation, test, split_check)
    if args.backend == "mock":
        if not all("fixture" in Path(p).name for p in (args.validation,args.test)):
            raise SystemExit("Mock backend can only evaluate named synthetic_fixture files")
        judge=MockJudge()
    else:
        if not args.model_id:
            raise SystemExit("Provide --model-id for local HF backend")
        judge=TransformersJudge(args.model_id,revision=_model_revision(args),dtype=args.dtype)
    val_scores,val_usage=run_sequences(validation,judge,args.mode,args.k,args.dense_model)
    threshold=calibrate_threshold(validation,val_scores,args.fpr_budget)
    test_scores,test_usage=run_sequences(test,judge,args.mode,args.k,args.dense_model)
    payload={
        "status":"SMOKE_TEST_ONLY" if args.backend=="mock" else "UNVALIDATED_PILOT",
        "history_mode":args.mode,"k":args.k,"judge":judge.name,"model_revision":args.model_revision,"dtype":args.dtype,
        "fpr_budget":args.fpr_budget,
        "threshold_fitted_on":"VALIDATION_ONLY",
        "split_integrity":split_check,
        "validation_summary":summarize(validation,val_scores,threshold),
        "test_summary":summarize(test,test_scores,threshold),
        "test_sequence_bootstrap_ci":{
            "timely_recall":bootstrap_sequence_ci(test,test_scores,threshold,"timely_recall",n_boot=200),
            "sequence_fpr":bootstrap_sequence_ci(test,test_scores,threshold,"sequence_fpr",n_boot=200),
        },
        "validation_usage":val_usage,"test_usage":test_usage,
        "warnings":["Synthetic mock scores are not research results." if args.backend=="mock" else "Pilot run, not reproduced paper baseline.",
                    "Sequence bootstrap is preliminary. Cluster by task family for publication.",
                    "Confirm harmful_index semantics, test family isolation, and published evaluator rules."],
    }
    _write_json(payload,args.out)

def _model_revision(args):
    """HF runs must pin a full model commit; otherwise refuse before any download."""
    from .paper import validate_dtype, validate_model_revision
    if not args.model_id:
        raise SystemExit("Provide --model-id for local HF backend")
    try:
        validate_dtype(args.dtype)
        return validate_model_revision(args.model_revision)
    except ValueError as error:
        raise SystemExit(str(error)) from None


def _paper_parts(args, data_paths):
    from .paper import (UPSTREAM_PROMPTS, MockPaperGenerator, TransformersGenerator,
                        load_upstream_template)
    template=load_upstream_template(args.paper_prompt, Path(args.upstream_prompts or UPSTREAM_PROMPTS))
    if args.backend == "mock":
        if not all("fixture" in Path(p).name for p in data_paths):
            raise SystemExit("Mock backend can only evaluate named synthetic_fixture files")
        return template, MockPaperGenerator()
    revision=_model_revision(args)
    return template, TransformersGenerator(args.model_id, revision, args.dtype,
                                           max_input_tokens=args.max_input_tokens)


def _smoke_sample(records, n):
    """Deterministic: per label, round-robin across task types, each ordered by sha256(id)."""
    import hashlib
    from itertools import chain, zip_longest
    picked=[]
    for label,quota in (("harmful",(n+1)//2),("benign",n//2)):
        by_type={}
        for rec in sorted((r for r in records if r.label==label),
                          key=lambda r:hashlib.sha256(r.sequence_id.encode()).hexdigest()):
            by_type.setdefault(rec.task_type,[]).append(rec)
        ordered=[r for r in chain.from_iterable(zip_longest(*by_type.values())) if r is not None]
        picked.extend(ordered[:quota])
    return picked


def _score_bucket(value):
    """Upstream takes the LAST number in a reply; values outside {0, 1} show the verdict
    may not be what was parsed (e.g. a trailing list index). Counts only, no text."""
    if value == 0:
        return "exactly_0"
    if value == 1:
        return "exactly_1"
    return "between_0_and_1" if 0 < value < 1 else "above_1"


def _history_eval(args):
    """Phase 3 stage runner. Test stage only reads a frozen threshold; it never fits one."""
    from .history import condition_name, run_stage
    from .paper import PaperJudge, validate_model_revision
    from .retrieval import DenseScorer
    if args.mode in ("current","full"):
        if args.k is not None:
            raise SystemExit(f"--k is meaningless for mode {args.mode}")
    elif args.k is None or args.k < 1:
        raise SystemExit("--k (>= 1) required for recent/bm25/dense/hybrid")
    name=condition_name(args.mode,args.k)
    split="validation" if args.stage=="validation" else "test"
    records=load_records(args.data,index_base=args.index_base,default_split=split,schema=args.schema)
    if args.ids:
        records=_restrict(records,args.ids,split)
    frozen=None
    if args.stage=="test":
        if not args.thresholds:
            raise SystemExit("Test stage requires --thresholds (frozen from validation)")
        table=json.loads(Path(args.thresholds).read_text())
        from .history import PROTOCOL_VERSION
        if table.get("protocol_version")!=PROTOCOL_VERSION:
            raise SystemExit(f"Frozen file is not protocol {PROTOCOL_VERSION}")
        if table.get("equivalence_margin_timely_recall") is None:
            raise SystemExit("Frozen file has no equivalence margin; set it before any test run")
        if name not in table["conditions"]:
            raise SystemExit(f"No frozen threshold for {name}")
        if table["fpr_budget"]!=args.fpr_budget:
            raise SystemExit("FPR budget differs from the frozen file")
        frozen=table["conditions"][name]["threshold"]
        frozen=float("inf") if frozen is None else frozen
    elif args.thresholds:
        raise SystemExit("Validation stage fits thresholds; do not pass --thresholds")
    dense=None
    if args.mode in ("dense","hybrid"):
        if not args.dense_model:
            raise SystemExit("--dense-model required for dense/hybrid")
        if args.backend=="hf":
            try:
                validate_model_revision(args.dense_revision)
            except ValueError as error:
                raise SystemExit(f"dense encoder: {error}") from None
        dense=DenseScorer(args.dense_model,revision=args.dense_revision)
    template,generator=_paper_parts(args,(args.data,))
    judge=PaperJudge(generator,template,args.paper_prompt,require_full_history=False)
    result=run_stage(records,judge,args.mode,args.k,dense,args.stage,args.fpr_budget,frozen)
    payload={
        "status":"SMOKE_TEST_ONLY" if args.backend=="mock" else "PHASE3_RUN",
        "judge":judge.name,"paper_prompt":args.paper_prompt,"model":generator.metadata(),
        "dense_encoder":{"model":args.dense_model,"revision":args.dense_revision} if dense else None,
        "ids_file":Path(args.ids).name if args.ids else None,
        "thresholds_file":Path(args.thresholds).name if args.thresholds else None,
        **result,
        "warnings":["Mock replies are not research results." if args.backend=="mock"
                    else "One condition of the Phase 3 comparison; compare only via the frozen plan.",
                    "Step 1 is not scored in any condition (paper protocol).",
                    "Readout A (primary): continuous first-token score, validation-calibrated threshold; "
                    "readout B: parsed score at fixed cutoff 1.0. Both use the >= firing rule; paper DSR uses >."],
    }
    _write_json(payload,args.out)


def _paper_smoke(args):
    """Score a few VALIDATION sequences in paper mode and report reply health only.

    No thresholds, no metrics, no test input. Reply text is never stored.
    """
    from collections import Counter
    from .dataset import _percentiles
    from .paper import PaperJudge, RecordingGenerator
    if args.n < 2:
        raise SystemExit("--n must be at least 2")
    records=load_records(args.validation,index_base=args.index_base,default_split="validation",
                         schema=args.schema)
    if args.ids:
        records=_restrict(records,args.ids,"validation")
    sample=_smoke_sample(records,args.n)
    template,inner=_paper_parts(args,(args.validation,))
    generator=RecordingGenerator(inner)
    judge=PaperJudge(generator,template,args.paper_prompt)
    statuses,overflow,errors,input_tokens=Counter(),0,Counter(),[]
    buckets=Counter()
    for rec in sample:
        try:
            scores,usage=run_sequences([rec],judge,"full",skip_first_turn=True)
        except ValueError as error:
            if "exceeds" in str(error):
                overflow+=1
            else:
                errors[type(error).__name__]+=1
            continue
        statuses.update(usage["parse_status_counts"])
        buckets.update(_score_bucket(v) for v in scores[rec.sequence_id][1:])
        if usage["input_tokens_total"] is not None and usage["decisions"]:
            input_tokens.append(usage["input_tokens_total"]//usage["decisions"])
    calls=sum(statuses.values())
    seconds=sorted(generator.call_seconds)
    payload={
        "status":"VALIDATION_SMOKE_NOT_A_RESULT" if args.backend=="hf" else "SMOKE_TEST_ONLY",
        "purpose":"Check real reply format, parsing, token sizes and latency before a full run.",
        "split":"validation","paper_prompt":args.paper_prompt,"model":inner.metadata(),
        "sequences_requested":args.n,"sequences_scored":len(sample)-overflow-sum(errors.values()),
        "sample_by_type_label":dict(sorted(Counter(f"{r.task_type}:{r.label}" for r in sample).items())),
        "sequences_context_overflow":overflow,"sequences_other_errors":dict(errors),
        "model_calls":calls,"parse_status_counts":dict(sorted(statuses.items())),
        "parse_ok_rate":statuses["ok"]/calls if calls else None,
        "parsed_score_buckets":dict(sorted(buckets.items())),
        "replies_over_5_tokens":sum(t>5 for t in generator.reply_tokens),
        "reply_chars_percentiles":_percentiles(generator.reply_chars),
        "reply_tokens_percentiles":_percentiles(generator.reply_tokens),
        "mean_input_tokens_per_call_by_sequence_percentiles":_percentiles(input_tokens),
        "seconds_per_call":{"mean":sum(seconds)/len(seconds) if seconds else None,
                            "p50":seconds[len(seconds)//2] if seconds else None,
                            "max":seconds[-1] if seconds else None},
        "warnings":["No metric or threshold is computed; do not report these numbers as findings.",
                    "Reply text is not stored; inspect parse_status_counts and reply sizes only."],
    }
    _write_json(payload,args.out)


def _per_sequence_record(rec, split, score_map, threshold):
    """IDs and numbers only (no request text), for paired comparisons across runs."""
    scores=score_map[rec.sequence_id][1:]  # step 1 is never scored in paper mode
    fired=None if threshold is None else next(
        (i+2 for i,v in enumerate(scores) if v>threshold), None)
    return {"id":rec.sequence_id,"split":split,"task_type":rec.task_type,"label":rec.label,
            "harm_step":rec.harmful_index+1 if rec.harmful_index is not None else None,
            "scores":[round(v,4) for v in scores],"first_fire_step":fired}


def _evaluate_paper(args, validation, test, split_check):
    """Paper protocol: threshold chosen by F1 on VALIDATION only, then one test pass."""
    from .metrics import paper_outcomes_by_task_type, paper_protocol_outcomes
    from .paper import PaperJudge, select_paper_threshold
    if args.mode != "full":
        raise SystemExit("Paper protocol uses cumulative context; use --mode full")
    template,generator=_paper_parts(args,(args.validation,args.test))
    judge=PaperJudge(generator, template, args.paper_prompt)
    val_scores,val_usage=run_sequences(validation,judge,"full",skip_first_turn=True)
    excluded=set(json.loads(Path(args.exclude).read_text())["union_ids"]) if args.exclude else None

    def block(val, tst):
        out={}
        for name,aligned in (("upstream_sweep",False),("aligned_sweep",True)):
            threshold=select_paper_threshold(val,val_scores,aligned=aligned)
            out[name]={"threshold_from_validation":threshold,
                       "validation":paper_protocol_outcomes(val,val_scores,threshold)
                       if threshold is not None else None,
                       "validation_by_task_type":paper_outcomes_by_task_type(val,val_scores,threshold)
                       if threshold is not None else None,
                       "test_rows":len(tst)}
        return out

    # Freeze every threshold before the single test pass.
    runs=[(block(validation,test),test)]
    if excluded is not None:
        kept_test=[r for r in test if r.sequence_id not in excluded]
        runs.append((block([r for r in validation if r.sequence_id not in excluded],kept_test),kept_test))
    primary,sensitivity=runs[0][0],(runs[1][0] if len(runs)>1 else None)
    test_scores,test_usage=run_sequences(test,judge,"full",skip_first_turn=True)
    for result,rows in runs:
        for entry in result.values():
            t=entry["threshold_from_validation"]
            entry["test"]=paper_protocol_outcomes(rows,test_scores,t) if t is not None else None
            entry["test_by_task_type"]=(paper_outcomes_by_task_type(rows,test_scores,t)
                                        if t is not None else None)
    primary_t=primary["upstream_sweep"]["threshold_from_validation"]
    per_sequence=[_per_sequence_record(r,split,scores,primary_t)
                  for split,rows,scores in (("validation",validation,val_scores),("test",test,test_scores))
                  for r in rows]
    payload={
        "status":"SMOKE_TEST_ONLY" if args.backend=="mock" else "UNVALIDATED_PAPER_MODE_RUN",
        "protocol":"PAPER_ITERATIVE","judge":judge.name,"paper_prompt":args.paper_prompt,
        "model":generator.metadata(),
        "threshold_fitted_on":"VALIDATION_ONLY","threshold_grid":"0.05..0.95 step 0.05",
        "primary_threshold_selector":"upstream_sweep",
        "split_integrity":split_check,"ids_file":Path(args.ids).name if args.ids else None,
        "primary_all_rows":primary,
        "sensitivity_exclusions":sensitivity,
        "exclude_file":Path(args.exclude).name if args.exclude else None,
        "validation_usage":val_usage,"test_usage":test_usage,
        "per_sequence_fields":"id, split, task_type, label, harm_step (1-based), "
                              "scores for steps 2..n, first_fire_step (1-based, primary threshold)",
        "per_sequence":per_sequence,
        "warnings":["Mock replies are not research results." if args.backend=="mock"
                    else "Local model run; compare with paper only after documenting deviations.",
                    "Unparsable replies are mapped to 0.0 as upstream does; see parse_status_counts.",
                    "upstream_sweep reproduces the paper's step-shifted threshold search."],
    }
    _write_json(payload,args.out)


if __name__=="__main__":
    main()
