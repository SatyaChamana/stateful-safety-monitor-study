"""Causal per-turn evaluation. Monitor never receives labels, future steps, or original goals."""
from __future__ import annotations
import time
from .retrieval import DenseScorer, select_history
from .schema import MonitorView, Sequence


def run_sequences(records: list[Sequence], judge, history_mode: str, k=3, dense_model=None,
                  skip_first_turn=False, continuous_out: dict | None = None):
    """skip_first_turn: paper protocol never scores step 1; its slot holds NaN, never a score.

    dense_model: a model name or a ready DenseScorer; built once here, never per turn.
    continuous_out: if given, filled with id -> per-turn Decision.continuous_score (NaN if absent).
    """
    masses=[]
    if isinstance(dense_model,str) and history_mode in ("dense","hybrid"):
        dense_model=DenseScorer(dense_model)
    scores={}
    usage={"decisions":0,"history_items":0,"input_tokens_total":0,"output_tokens_total":0,
           "token_count_available":True,"elapsed_seconds":0.0,"turns_not_scored":0,
           "parse_status_counts":{},"retrieval_seconds":0.0,
           "decisions_history_shorter_than_full":0}
    for rec in records:
        out=[]
        cont=[]
        for t, current in enumerate(rec.steps):
            if skip_first_turn and t==0:
                out.append(float('nan'))
                cont.append(float('nan'))
                usage['turns_not_scored']+=1
                continue
            prior=[(i,rec.steps[i]) for i in range(t)]
            started=time.perf_counter()
            selected=select_history(history_mode,current,prior,k,dense=dense_model)
            usage['retrieval_seconds']+=time.perf_counter()-started
            if any(index>=t for index,_ in selected):
                raise AssertionError("Temporal leakage detected")
            decision=judge.score(MonitorView(sequence_id=rec.sequence_id,turn_index=t,
                                             current_request=current,history=tuple(selected)))
            out.append(float(decision.score))
            cont.append(float('nan') if decision.continuous_score is None else float(decision.continuous_score))
            if decision.readout_mass is not None:
                masses.append(decision.readout_mass)
            if decision.parse_status is not None:
                counts=usage['parse_status_counts']
                counts[decision.parse_status]=counts.get(decision.parse_status,0)+1
            usage['decisions']+=1
            usage['history_items']+=len(selected)
            usage['decisions_history_shorter_than_full']+=len(selected)<t
            usage['elapsed_seconds']+=decision.elapsed_seconds or 0
            if decision.input_tokens is None:
                usage['token_count_available']=False
            else:
                usage['input_tokens_total']+=decision.input_tokens
            if decision.output_tokens is not None:
                usage['output_tokens_total']+=decision.output_tokens
        scores[rec.sequence_id]=out
        if continuous_out is not None:
            continuous_out[rec.sequence_id]=cont
    if not usage['token_count_available']:
        usage['input_tokens_total']=None
        usage['output_tokens_total']=None
    if masses:
        ordered=sorted(masses)
        usage['readout_mass']={"n":len(ordered),"min":ordered[0],"p10":ordered[len(ordered)//10],
                               "median":ordered[len(ordered)//2],"below_0_5":sum(m<0.5 for m in ordered)}
    usage['mean_history_items']=usage['history_items']/usage['decisions'] if usage['decisions'] else None
    return scores, usage
