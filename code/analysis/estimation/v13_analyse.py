"""Analysis of fixed-origin route outputs; run on server only.

Input columns (long): city,uid,stratum,pop,design_weight,N_h,n_h,scope,mode,
profile,time_min,status; optional finite_unknown (technical search failures).
Coverage refers to returned candidate routes, not observed clinical access.
"""
from pathlib import Path
import argparse, json, math
import numpy as np
import pandas as pd

TRAPEZOID = getattr(np, "trapezoid", np.trapz)


def ratio_estimate(d, y):
    w = d.design_weight.to_numpy(float)
    y = np.asarray(y, dtype=float)
    good = np.isfinite(w) & np.isfinite(y) & (w > 0)
    d = d.loc[good].reset_index(drop=True)
    w, y = w[good], y[good]
    if not len(w):
        return dict(estimate=np.nan,se=np.nan,lower=np.nan,upper=np.nan)
    den = w.sum()
    # Preserve exact constants: the weighted dot product can round a constant
    # outcome (especially all ones) slightly away from its exact value.
    mu = float(y[0]) if np.all(y == y[0]) else np.dot(w, y)/den
    var = 0.0
    for _, ix in d.groupby('stratum', observed=True).groups.items():
        ix = np.array(list(ix))
        sub = d.iloc[ix]
        nh = len(ix)
        Nh = float(sub.N_h.iloc[0])
        if nh >= Nh:
            continue
        if nh < 2:
            return dict(estimate=100*mu,se=np.nan,lower=np.nan,upper=np.nan)
        # Stratified SRS linearization of a population-weighted ratio estimator.
        e = sub['pop'].to_numpy(float) * (y[ix]-mu)
        var += Nh**2 * (1-nh/Nh) * np.var(e, ddof=1)/nh
    se = math.sqrt(max(0,var))/den
    census=all(len(ix)>=float(d.loc[list(ix),'N_h'].iloc[0])
               for ix in d.groupby('stratum', observed=True).groups.values())
    suppress_interval=census or se==0
    return dict(estimate=100*mu,se=100*se,
                lower=np.nan if suppress_interval else 100*(mu-1.96*se),
                upper=np.nan if suppress_interval else 100*(mu+1.96*se))


def coverage_vector(d, tau):
    t = pd.to_numeric(d.time_min,errors='coerce').to_numpy(float)
    covered = (np.isfinite(t) & (t<=tau)).astype(float)
    # A technical failure can conceal a faster route even if another OD returned.
    if 'finite_unknown' in d:
        unknown = d.finite_unknown.fillna(False).astype(bool).to_numpy()
    else:
        unknown = d.status.isin(['technical_error','partial_error','not_queried']).to_numpy()
    upper = np.maximum(covered, unknown.astype(float))
    return covered,upper


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--times',required=True)
    ap.add_argument('--out',required=True)
    args=ap.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(args.times,dtype={'uid':str})
    df['stratum']=df['stratum'].fillna('unknown').astype(str)
    keys=['city','uid','scope','mode','profile']
    if df.duplicated(keys).any():
        raise ValueError('duplicate origin-arm rows')
    for col in ['N_h','n_h','design_weight','pop']:
        df[col]=pd.to_numeric(df[col],errors='raise')
    if (df.design_weight<=0).any() or (df['pop']<=0).any():
        raise ValueError('nonpositive population/weight')
    curves=[];grad=[];interaction=[];auc=[]
    thresholds=list(range(10,61,5))
    for arm,d in df.groupby(['city','scope','mode','profile'],sort=True):
        d=d.reset_index(drop=True)
        city,scope,mode,profile=arm
        census=all(len(g)>=float(g.N_h.iloc[0]) for _,g in d.groupby('stratum',observed=True))
        base=dict(city=city,scope=scope,mode=mode,profile=profile,n=len(d),
                  sampling_design='census' if census else 'stratified_SRS',
                  sampled_population=float(d['pop'].sum()),
                  estimated_population=float(d.design_weight.sum()))
        poor='Q1' if city=='Sao Paulo' else 'low'
        rich='Q5' if city=='Sao Paulo' else 'high'
        vals=[]
        for tau in thresholds:
            lo,hi=coverage_vector(d,tau)
            z=ratio_estimate(d,lo)
            up=ratio_estimate(d,hi)['estimate']
            curves.append(dict(**base,threshold=tau,coverage_lower=z['estimate'],
                               coverage_upper=up,design_se=z['se'],
                               design_ci_lower=max(0,z['lower']) if np.isfinite(z['lower']) else np.nan,
                               design_ci_upper=min(100,z['upper']) if np.isfinite(z['upper']) else np.nan))
            vals.append((tau,z['estimate'],up))
            ps=d.stratum.eq(poor);rs=d.stratum.eq(rich)
            if ps.any() and rs.any():
                p=ratio_estimate(d[ps],lo[ps]);r=ratio_estimate(d[rs],lo[rs])
                ph=ratio_estimate(d[ps],hi[ps]);rh=ratio_estimate(d[rs],hi[rs])
                se=math.hypot(p['se'],r['se'])
                g=p['estimate']-r['estimate']
                grad.append(dict(**base,threshold=tau,poor=poor,rich=rich,
                                 gradient=g,missing_bound_lower=p['estimate']-rh['estimate'],
                                 missing_bound_upper=ph['estimate']-r['estimate'],
                                 design_se=se,design_ci_lower=np.nan if census or se==0 else g-1.96*se,
                                 design_ci_upper=np.nan if census or se==0 else g+1.96*se))
        arr=np.array(vals)
        auc.append(dict(**base,range_min='10-60',
                        average_coverage_lower=TRAPEZOID(arr[:,1],arr[:,0])/50,
                        average_coverage_upper=TRAPEZOID(arr[:,2],arr[:,0])/50))
    # Joint contrasts preserve pairing and use one design variance, rather than
    # comparing significance labels of four separate coverage estimates.
    for (city,profile),dd in df.groupby(['city','profile'],sort=True):
        for scope2 in ['E','A']:
            parts=[]
            for scope,mode,sign in [('R','wt',1),(scope2,'wt',-1),('R','car',-1),(scope2,'car',1)]:
                z=dd[(dd.scope==scope)&(dd['mode']==mode)].copy()
                if z.empty:break
                z=z.set_index('uid')
                parts.append((z,sign))
            if len(parts)!=4:continue
            idx=parts[0][0].index
            if any(set(z.index)!=set(idx) for z,_ in parts):
                raise ValueError('interaction arms do not share complete origin sets')
            base=parts[0][0].loc[idx].reset_index()
            for tau in thresholds:
                y=np.zeros(len(idx));blo=np.zeros(len(idx));bhi=np.zeros(len(idx))
                for z,sign in parts:
                    lo,hi=coverage_vector(z.loc[idx].reset_index(),tau)
                    y+=sign*lo
                    blo+=sign*(lo if sign>0 else hi)
                    bhi+=sign*(hi if sign>0 else lo)
                r=ratio_estimate(base,y)
                interaction.append(dict(city=city,profile=profile,scope1='R',scope2=scope2,
                                        sampling_design='census' if all(len(g)>=float(g.N_h.iloc[0]) for _,g in base.groupby('stratum',observed=True)) else 'stratified_SRS',
                                        threshold=tau,contrast=r['estimate'],design_se=r['se'],
                                        design_ci_lower=r['lower'],design_ci_upper=r['upper'],
                                        missing_bound_lower=ratio_estimate(base,blo)['estimate'],
                                        missing_bound_upper=ratio_estimate(base,bhi)['estimate']))
    pd.DataFrame(curves).to_csv(out/'coverage_curves.csv',index=False)
    pd.DataFrame(grad).to_csv(out/'income_gradients.csv',index=False)
    pd.DataFrame(interaction).to_csv(out/'mode_scope_contrasts.csv',index=False)
    pd.DataFrame(auc).to_csv(out/'threshold_area_summaries.csv',index=False)
    quality=df.groupby(['city','scope','mode','profile','status']).agg(
        n=('uid','size'),population=('pop','sum'),design_population=('design_weight','sum')).reset_index()
    quality.to_csv(out/'route_status_population.csv',index=False)
    # Spatial source data retain only de-identified geographic origins and weights.
    s=df[df.profile.astype(str).str.startswith('baseline')].copy()
    s['covered30']=pd.to_numeric(s.time_min,errors='coerce').le(30)
    s.to_csv(out/'spatial_origin_arms.csv',index=False)
    (out/'ANALYSIS_NOTES.json').write_text(json.dumps(dict(
        independent_unit='sampled geographic origin',
        coverage='conditional on queried OTP candidate routes and input network',
        design='stratified SRS ratio linearization with finite-population correction; census strata variance zero',
        ci='pointwise design interval for modeled lower estimate; not a model/input uncertainty interval',
        missing='technical failures propagated to bounds, no_pattern is separately labelled modeled search outcome',
        thresholds=thresholds,area='trapezoidal integral divided by 50 minutes',
        contrast='descriptive additive mode-by-scope contrast, not causal difference-in-differences',
        multiplicity='full pre-specified curves; no per-threshold discovery claims',
        n_rows=len(df)),indent=2),encoding='utf-8')
    print('DONE',len(curves),'coverage rows',len(interaction),'contrasts',flush=True)


if __name__=='__main__':main()
