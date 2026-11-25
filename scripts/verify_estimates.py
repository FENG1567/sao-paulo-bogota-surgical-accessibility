"""Compare recomputed primary estimates to the frozen publication sources."""
from pathlib import Path
import argparse,csv,json,math
ROOT=Path(__file__).resolve().parents[1]
TOLERANCE=1e-10
def load(path,keys):
    with path.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    out={}
    for row in rows:
        key=tuple(str(float(row[k])) if k=='threshold' else row[k] for k in keys)
        if key in out: raise ValueError(f'Duplicate key in {path.name}: {key}')
        out[key]=row
    return out
def num(value):
    if value is None or str(value).strip().lower() in {'','nan','na','none','null'}:return None
    out=float(value)
    if not math.isfinite(out):raise ValueError('Unexpected nonfinite numerical result')
    return out
def compare(actual_path,frozen_path,keys,columns,subset=False):
    actual=load(actual_path,keys);frozen=load(frozen_path,keys)
    if subset:actual={k:v for k,v in actual.items() if k in frozen}
    if actual.keys()!=frozen.keys():raise AssertionError(f'Key mismatch: {frozen_path.name}')
    details={}
    for col in columns:
        finite=missing=0;maximum=0.0
        for key in frozen:
            a,b=num(actual[key][col]),num(frozen[key][col])
            if (a is None)!=(b is None):raise AssertionError(f'Missingness differs: {frozen_path.name} {key} {col}')
            if a is None:missing+=1;continue
            finite+=1;delta=abs(a-b);maximum=max(maximum,delta)
            if delta>TOLERANCE:raise AssertionError(f'Estimate differs: {frozen_path.name} {key} {col}: absolute error {delta}')
        details[col]={'finite_pairs':finite,'both_missing':missing,'max_absolute_error':maximum}
    return {'status':'PASS','matched_rows':len(frozen),'columns':details}
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--estimate-dir',type=Path,default=ROOT/'artifacts/estimation')
    args=parser.parse_args();d=args.estimate_dir
    keys=['city','scope','mode','profile','threshold']
    coverage=['coverage_lower','coverage_upper','design_se','design_ci_lower','design_ci_upper']
    report={
     'Table2':compare(d/'coverage_curves.csv',ROOT/'data/source_data/Table2_main_15_30_45_source.csv',keys,coverage+['sampled_population','estimated_population','n'],True),
     'Figure2_curve':compare(d/'coverage_curves.csv',ROOT/'data/source_data/Figure2_main_curve_source.csv',keys,coverage),
     'Figure2_paired':compare(d/'mode_scope_contrasts.csv',ROOT/'data/source_data/Figure2_paired_scope_contrast_source.csv',['city','profile','scope1','scope2','threshold'],['contrast','design_se','design_ci_lower','design_ci_upper','missing_bound_lower','missing_bound_upper'])
    }
    print(json.dumps({'status':'PASS','absolute_tolerance':TOLERANCE,'comparisons':report},indent=2))
if __name__=='__main__':main()
