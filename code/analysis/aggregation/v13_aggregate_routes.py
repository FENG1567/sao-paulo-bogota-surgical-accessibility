"""Aggregate candidate-level JSONL, preserving the pre-routed origin frame."""
import argparse,json,math,hashlib,csv,sys
from pathlib import Path
import numpy as np
import pandas as pd

TECHNICAL={'timeout','http_error','connection_error','graphql_error','pending','config_error'}
QUERY_IDENTITY_PROTOCOL='route-query-identity-v1'
COORD_TOL=1e-10

def file_sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def coord(value, label, low, high):
    if value is None or isinstance(value, bool):raise ValueError(f'{label} is missing or invalid')
    try:x=float(value)
    except (TypeError,ValueError) as exc:raise ValueError(f'{label} is missing or invalid') from exc
    if not math.isfinite(x) or not low<=x<=high:raise ValueError(f'{label} must be finite and in [{low},{high}]')
    return x


def od_hex(origin, destination, label='route'):
    if not isinstance(origin,dict) or not isinstance(destination,dict):raise ValueError(f'{label} coordinates are missing')
    o=(coord(origin.get('lat'),f'{label} origin latitude',-90,90),coord(origin.get('lon'),f'{label} origin longitude',-180,180))
    d=(coord(destination.get('lat'),f'{label} destination latitude',-90,90),coord(destination.get('lon'),f'{label} destination longitude',-180,180))
    return (o[0].hex(),o[1].hex(),d[0].hex(),d[1].hex()),o,d


def strict_flag(value,label):
    if isinstance(value,bool):return value
    if value is None:raise ValueError(f'{label} is missing')
    if isinstance(value,(int,float)) and value in (0,1):return bool(value)
    text=str(value).strip().casefold()
    if text in {'0','false','f','no','n',''}:return False
    if text in {'1','true','t','yes','y'}:return True
    raise ValueError(f'{label} is not a boolean flag')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--origins',required=True)
    ap.add_argument('--facilities',required=True)
    ap.add_argument('--routes',required=True,nargs='+')
    ap.add_argument('--out',required=True)
    ap.add_argument('--profile',default='baseline')
    ap.add_argument('--modes',default='car,wt,foot')
    ap.add_argument('--car-factor',type=float,default=1/0.60)
    ap.add_argument('--caps',action='store_true')
    ap.add_argument('--config',help='require exact frozen request hashes, coordinates, flags and candidate completeness')
    ap.add_argument('--k',type=int,help='destination pre-screen k used for this materialized run')
    ap.add_argument('--query-identity-protocol')
    ap.add_argument('--query-identity-approval-sha256')
    ap.add_argument('--query-identity-module-sha256')
    args=ap.parse_args()
    identity_args=(args.query_identity_protocol,args.query_identity_approval_sha256,args.query_identity_module_sha256)
    if any(identity_args) and (not all(identity_args) or args.query_identity_protocol!=QUERY_IDENTITY_PROTOCOL):
        raise ValueError('incomplete or mismatched approved query-identity execution metadata')
    origins=pd.read_csv(args.origins,dtype={'uid':str})
    origins['stratum']=origins['stratum'].fillna('unknown')
    if origins.uid.duplicated().any():raise ValueError('duplicate uid in origin frame')
    expected=set(origins.uid)
    facilities=pd.read_csv(args.facilities,dtype={'site_id':str})
    def city_key(x):return str(x).strip().replace('_',' ').casefold()
    scope_available={}
    for city,g in facilities.groupby('city'):
        scope_available[city_key(city)]={s for s in ['R','E','A']
            if s in g and pd.to_numeric(g[s],errors='coerce').fillna(0).gt(0).any()}
    if any(not scope_available.get(city_key(city)) for city in origins.city.unique()):
        raise ValueError('a target city has no declared facility scope; require explicit facilities')
    origin_city=origins.set_index('uid').city.map(city_key).to_dict()
    origin_truth=origins.set_index('uid').to_dict('index')
    facilities['city_key']=facilities.city.map(city_key)
    facility_truth={}
    for row in facilities.to_dict('records'):
        key=(row['city_key'],str(row['site_id']))
        if key in facility_truth:raise ValueError('duplicate facility city/site')
        facility_truth[key]=row
    strict=bool(args.config)
    expected_routes=set();actual_routes=set()
    if strict:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "routing"))
        from v13_route_engine import choose_candidates,route_param_hash
        config=json.loads(Path(args.config).read_text())
        kval=args.k or int(config.get('k',8))
        with open(args.facilities,encoding='utf-8-sig',newline='') as handle:
            candidate_facilities=list(csv.DictReader(handle))
        hashes={(city_key(c),m):route_param_hash(config,m,c) for c in origins.city.unique() for m in args.modes.split(',')}
        for o in origins.to_dict('records'):
            for f in choose_candidates(o,candidate_facilities,kval):
                for m in args.modes.split(','):expected_routes.add((city_key(o['city']),str(o['uid']),str(f['site_id']),m,hashes[city_key(o['city']),m]))
    records={};logical_seen=set();physical_seen=set();application_seen=set();physical_signatures={}
    for path in args.routes:
        with open(path,encoding='utf-8') as f:
            for line in f:
                r=json.loads(line)
                if r.get('record_type')!='route' or r.get('uid') not in expected:continue
                uid=str(r['uid']);mode=r.get('mode');site=str(r.get('site_id'));rh=r.get('request_hash')
                if city_key(r['city'])!=origin_city[uid]:
                    raise ValueError('route city contradicts the fixed origin frame')
                ck=city_key(r['city'])
                logical_key=(ck,uid,site,mode,rh)
                if mode not in args.modes.split(','):raise ValueError('unexpected route mode')
                if not isinstance(rh,str) or not rh.strip():raise ValueError('route request hash missing')
                ft=facility_truth.get((ck,site))
                if ft is None:raise ValueError('route site absent from frozen facilities')
                if uid not in origin_truth:raise ValueError('route UID absent from fixed origin frame')
                od,rc_origin,rc_dest=od_hex(r.get('origin'),r.get('destination'),f'route {uid}/{site}/{mode}')
                truth_origin=origin_truth[uid]
                for got,truth,label in ((rc_origin[0],coord(truth_origin['lat'],'origin truth latitude',-90,90),'origin latitude'),
                                        (rc_origin[1],coord(truth_origin['lon'],'origin truth longitude',-180,180),'origin longitude'),
                                        (rc_dest[0],coord(ft['lat'],'facility truth latitude',-90,90),'destination latitude'),
                                        (rc_dest[1],coord(ft['lon'],'facility truth longitude',-180,180),'destination longitude')):
                    if abs(got-truth)>COORD_TOL:raise ValueError(f'route {label} mismatch')
                if strict:
                    if logical_key not in expected_routes:raise ValueError('unexpected origin/site/mode/hash in route input')
                    if rh!=hashes[city_key(r['city']),mode]:raise ValueError('route request hash contradicts frozen configuration')
                flags=r.get('facility_flags')
                if not isinstance(flags,dict) or set(flags)!={'R','E','A'}:raise ValueError('route facility flags missing, malformed, or contain unexpected scopes')
                normalized_flags={}
                for s in ['R','E','A']:
                    truth=bool(float(ft.get(s,0))) if pd.notna(ft.get(s,0)) else False
                    normalized_flags[s]=strict_flag(flags.get(s),f'route {uid}/{site} {s}')
                    if normalized_flags[s]!=truth:raise ValueError('route flags contradict frozen site membership')
                r['facility_flags']=normalized_flags
                st=r.get('status');patterns=r.get('patterns')
                if st not in TECHNICAL|{'success','no_pattern'} or not isinstance(patterns,list):raise ValueError('invalid route status or pattern payload')
                if st=='success' and not patterns:raise ValueError('success route has no returned patterns')
                if st!='success' and patterns:raise ValueError('non-success route unexpectedly carries patterns')
                for p in patterns:
                    if not isinstance(p,dict) or not isinstance(p.get('legs'),list):raise ValueError('malformed returned pattern')
                    for value,label in ((p.get('duration'),'duration'),(p.get('foot_distance',0),'foot distance'),
                                        (p.get('journey_duration_min',p.get('duration')),'journey duration'),
                                        (p.get('initial_departure_gap_min',0),'initial departure gap')):
                        try:v=float(value)
                        except (TypeError,ValueError) as exc:raise ValueError(f'invalid pattern {label}') from exc
                        if not math.isfinite(v) or v<0:raise ValueError(f'invalid pattern {label}')
                    for leg in p['legs']:
                        if not isinstance(leg,dict) or not isinstance(leg.get('mode'),str):raise ValueError('malformed returned route leg')
                        for value,label in ((leg.get('duration',0),'leg duration'),(leg.get('distance',0),'leg distance')):
                            try:v=float(value)
                            except (TypeError,ValueError) as exc:raise ValueError(f'invalid {label}') from exc
                            if not math.isfinite(v) or v<0:raise ValueError(f'invalid {label}')
                actual_routes.add(logical_key)
                logical_seen.add(logical_key)
                physical_key=(ck,*od,mode,rh)
                application_key=(logical_key,od)
                st=r['status']
                signature=hashlib.sha256(json.dumps({'status':st,'patterns':r.get('patterns',[])},sort_keys=True,separators=(',',':'),allow_nan=False).encode()).digest()
                if physical_key in physical_signatures and physical_signatures[physical_key]!=signature:
                    raise ValueError(f'conflicting payload for exact physical query {uid}/{site}/{mode}; rematerialize latest cache')
                physical_signatures[physical_key]=signature
                physical_seen.add(physical_key)
                if application_key in application_seen:
                    continue
                application_seen.add(application_key)
                for scope,flag in r['facility_flags'].items():
                    if not flag:continue
                    if scope not in scope_available.get(city_key(r['city']),set()):
                        raise ValueError('route flags contradict declared facility scopes')
                    profiles=[(mode,None)]
                    if args.caps and mode=='wt':profiles+= [('wt_cap1km',1000),('wt_cap2km',2000),('wt_cap3km',3000)]
                    # A separately returned direct walk is a feasible WT option,
                    # even if OTP's finite transit-pattern list omitted it.
                    if mode=='foot':
                        profiles.append(('wt',None))
                        if args.caps:profiles+= [('wt_cap1km',1000),('wt_cap2km',2000),('wt_cap3km',3000)]
                    for arm,cap in profiles:
                        ak=(uid,scope,arm)
                        z=records.setdefault(ak,dict(best=math.inf,best_site=None,
                            best_walk_m=np.nan,best_modes=None,best_journey=math.inf,
                            winning_journey=np.nan,winning_initial_wait=np.nan,
                            technical=0,no_pattern=0,success=0,queried=0,
                            queried_sites=set(),success_sites=set(),technical_sites=set(),no_pattern_sites=set()))
                        z['queried']+=1
                        z['queried_sites'].add(site)
                        if st in TECHNICAL:
                            z['technical']+=1;z['technical_sites'].add(site);continue
                        if st=='no_pattern':z['no_pattern']+=1;z['no_pattern_sites'].add(site);continue
                        if st!='success':
                            z['technical']+=1;z['technical_sites'].add(site);continue
                        z['success']+=1
                        z['success_sites'].add(site)
                        for p in r['patterns']:
                            t=float(p['duration']);walk=float(p.get('foot_distance',0))
                            if not math.isfinite(t) or t<0 or not math.isfinite(walk) or walk<0:raise ValueError('invalid route duration or walk distance')
                            if cap is not None and walk>cap+1e-6:continue
                            journey=float(p.get('journey_duration_min',t))
                            if not math.isfinite(journey) or journey<0:raise ValueError('invalid journey duration')
                            z['best_journey']=min(z['best_journey'],journey)
                            tie_key=(str(site),json.dumps(p,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False))
                            if t<z['best'] or (t==z['best'] and (z.get('_winner_tie') is None or tie_key<z['_winner_tie'])):
                                z.update(best=t,best_site=site,best_walk_m=walk,
                                    winning_journey=journey,winning_initial_wait=float(p.get('initial_departure_gap_min',0)),
                                    best_modes='+'.join(str(l['mode']) for l in p['legs']),_winner_tie=tie_key)
    if strict and actual_routes!=expected_routes:
        raise ValueError(f'incomplete materialized candidate file: missing={len(expected_routes-actual_routes)}')
    rows=[]
    modes=args.modes.split(',')
    if args.caps and 'wt' in modes:modes+=['wt_cap1km','wt_cap2km','wt_cap3km']
    for _,o in origins.iterrows():
        meta={k:o[k] for k in ['city','uid','lat','lon','stratum','pop','design_weight','N_h','n_h','inclusion_prob']}
        for scope in sorted(scope_available.get(city_key(o.city),set())):
            for mode in modes:
                z=records.get((o.uid,scope,mode),{})
                t=z.get('best',math.inf)
                error=z.get('technical',0)>0
                if not z:status='not_queried';error=True
                elif error:status='partial_error' if math.isfinite(t) else 'technical_error'
                elif math.isfinite(t):status='ok'
                else:status='no_returned_feasible_candidate'
                if mode=='car' and math.isfinite(t):t*=args.car_factor
                rows.append(dict(**meta,scope=scope,mode=mode,profile=args.profile,
                    time_min=t if math.isfinite(t) else np.nan,status=status,finite_unknown=error,
                    best_journey_min=(z.get('best_journey',np.nan)*(args.car_factor if mode=='car' else 1)) if math.isfinite(z.get('best_journey',math.inf)) else np.nan,
                    winning_journey_min=z.get('winning_journey',np.nan)*(args.car_factor if mode=='car' else 1),
                    winning_initial_wait_min=z.get('winning_initial_wait',np.nan),
                    best_site=z.get('best_site'),best_modes=z.get('best_modes'),
                    best_walk_m=z.get('best_walk_m'),queried_destinations=len(z.get('queried_sites',set())),
                    queried_search_profiles=z.get('queried',0),
                    successful_destinations=len(z.get('success_sites',set())),
                    no_pattern_destinations=len(z.get('no_pattern_sites',set())),technical_destinations=len(z.get('technical_sites',set()))))
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    result=pd.DataFrame(rows)
    if result.empty:raise ValueError('no routes matched the fixed origins; do not publish empty coverage')
    result.to_csv(out,index=False)
    # Returned-candidate caps must be nested when every cap uses the same pool.
    if args.caps:
        p=result.pivot(index=['city','uid','scope'],columns='mode',values='time_min')
        for tight,loose in [('wt_cap1km','wt_cap2km'),('wt_cap2km','wt_cap3km'),('wt_cap3km','wt')]:
            if tight not in p or loose not in p:continue
            bad=p[tight].notna() & (p[loose].isna() | (p[tight]+1e-8<p[loose]))
            if bad.any():raise ValueError(f'non-nested common-pool caps: {tight}/{loose}, n={bad.sum()}')
    out.with_suffix('.notes.json').write_text(json.dumps(dict(
        n_origin=len(origins),n_arm_rows=len(result),n_unique_queries=len(application_seen),
        query_identity={
            'protocol_version':QUERY_IDENTITY_PROTOCOL,
            'logical_completeness_key':['city','uid','site_id','mode','request_hash'],
            'physical_payload_key':['city','origin_lat_hex','origin_lon_hex','destination_lat_hex','destination_lon_hex','mode','request_hash'],
            'application_dedup_key':['logical_completeness_key','OD_float_hex'],
            'n_logical_keys':len(logical_seen),'n_physical_payload_keys':len(physical_seen),
            'n_application_keys':len(application_seen),
            'logical_keys_with_multiple_physical_variants':sum(n>1 for n in __import__('collections').Counter(k for k,od in application_seen).values()),
            'physical_variants_beyond_first':sum(max(0,n-1) for n in __import__('collections').Counter(k for k,od in application_seen).values()),
            'coordinate_tolerance_degrees':COORD_TOL,
            'approval_sha256':args.query_identity_approval_sha256,
            'scoped_module_sha256':args.query_identity_module_sha256,
        },
        car_factor=args.car_factor,profile=args.profile,
        scope_minimum='minimum over union of queried scope-nearest destinations belonging to this scope',
        walking_caps='filtered common returned candidate pool; not a proof of global constrained optimality',
        technical_failures='not discarded, propagated to coverage bounds',
        wt_definition='union of WT returned patterns and separately queried pure-foot patterns on the same physical OD pool',
        no_pattern='no returned route, interpreted only under modeled query/network conditions',
        route_files=args.routes),indent=2),encoding='utf-8')
    notes_path=out.with_suffix('.notes.json')
    notes=json.loads(notes_path.read_text())
    notes['provenance']={
        'aggregator_code_sha256':file_sha(__file__),
        'origins_sha256':file_sha(args.origins),'facilities_sha256':file_sha(args.facilities),
        'config_sha256':file_sha(args.config) if args.config else None,
        'route_files_sha256':{str(p):file_sha(p) for p in args.routes},
        'wt_union_version':'WT-returned-patterns-plus-independent-foot-v1'}
    notes_path.write_text(json.dumps(notes,indent=2),encoding='utf-8')
    print('DONE',len(result),'origin-arm rows',flush=True)


if __name__=='__main__':main()
