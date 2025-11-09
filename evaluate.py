from geoecgan.metrics import cer, bleu

def evaluate_chinese(refs, hyps):
    assert len(refs) == len(hyps)
    cer_vals = [cer(r, h) for r, h in zip(refs, hyps)]
    bleu_vals = [bleu(r, h, max_n=4) for r, h in zip(refs, hyps)]
    return {"CER(%)": sum(cer_vals)/max(1,len(cer_vals)), "BLEU": sum(bleu_vals)/max(1,len(bleu_vals))}
