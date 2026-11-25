"""Check release integrity, data schemas, publication mappings and optional outputs."""
from pathlib import Path
import argparse, csv, hashlib, json, sys

ROOT = Path(__file__).resolve().parents[1]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--check-artifacts',action='store_true')
    args=ap.parse_args(); checks=[]
    def check(condition,message):
        if not condition: raise AssertionError(message)
        checks.append(message)
    manifest=json.loads((ROOT/'metadata/release_manifest.json').read_text())
    for item in manifest['files']:
        p=ROOT/item['path']
        check(p.is_file(),f"file exists: {item['path']}")
        check(sha(p)==item['sha256'],f"SHA-256 matches: {item['path']}")
    check(not any(ROOT.rglob('*.ttf')),'no bundled TrueType fonts')
    check(not any(ROOT.rglob('*.otf')),'no bundled OpenType fonts')
    schema=json.loads((ROOT/'metadata/data_dictionary.json').read_text())['files']
    for item in schema:
        with (ROOT/item['path']).open(encoding='utf-8-sig',newline='') as f:
            reader=csv.reader(f); header=next(reader); rows=list(reader)
        check(header==item['columns'],f"column order: {item['path']}")
        check(len(rows)==item['rows'],f"row count: {item['path']}")
        check(all(len(r)==len(header) for r in rows),f"rectangular CSV: {item['path']}")
    tables=json.loads((ROOT/'metadata/table_crosswalk.json').read_text())['tables']
    check(len(tables)==18,'18 table input files')
    check(sum(t['role']=='publication' for t in tables)==17,'17 publication table displays')
    check(len({t['output'] for t in tables})==18,'unique table output names')
    figures=json.loads((ROOT/'metadata/figure_crosswalk.json').read_text())['figures']
    check(len(figures)==8,'eight publication figures mapped')
    for name, inputs in figures.items():
        check((ROOT/'reference/figures'/name).exists(),f'reference present: {name}')
        check(bool(inputs),f'nonempty source mapping: {name}')
        for src in inputs: check((ROOT/'data/source_data'/src).exists(),f'figure source present: {src}')
    if args.check_artifacts:
        import fitz
        panel_count=0
        for name in figures:
            p=ROOT/'artifacts/figures'/name
            check(p.is_file(),f'redrawn PDF exists: {name}')
            with fitz.open(p) as doc:
                check(doc.page_count==1,f'one figure page: {name}')
                blocks=doc[0].get_text('dict')['blocks']
                letters=[s for b in blocks if b.get('type')==0 for line in b['lines'] for s in line['spans']
                         if s['text'].strip() in list('ABCDEFGH') and abs(s['size']-18)<0.12]
                check(bool(letters),f'18-point panel letters: {name}')
                check(all('Arial' in s['font'] and 'Bold' in s['font'] for s in letters),f'Arial Bold panel letters: {name}')
                panel_count+=len(letters)
        check(panel_count==32,'32 panel letters across all eight figures')
        for table in tables:
            src=ROOT/'data/tables'/table['source']
            dst=ROOT/'artifacts/tables'/(table['output']+'.csv')
            check(dst.exists(),f'table export exists: {dst.name}')
            with src.open(encoding='utf-8-sig',newline='') as f: original=list(csv.reader(f))
            with dst.open(encoding='utf-8-sig',newline='') as f: exported=list(csv.reader(f))
            check(original==exported,f'all table cells preserved: {dst.name}')
    print(json.dumps({'status':'PASS','checks':len(checks),'artifact_checks':args.check_artifacts},indent=2))
if __name__=='__main__':
    try: main()
    except Exception as e:
        print(f'FAIL: {e}',file=sys.stderr);sys.exit(1)
