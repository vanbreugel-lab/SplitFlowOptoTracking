"""
Smoke test: can every script and notebook in the repository find the data it asks for?

Reads all tracked .py files and all notebook code cells WITHOUT running them, and

  1. collects every data lookup:  data_path('key', ...), data_dir('key'), result_path('file name'),
     trajectory_for_result('file name'), and every key or results file name kept in a table or a variable
  2. checks that the key (or results file name) is listed in Data/data_paths.yaml and that the file or
     folder it points to exists
  3. reports any hard-coded data path that bypasses the config, and any lookup function a notebook uses
     in a cell that comes before the cell importing it

It covers the code the RunFigures/ scripts never run (fit scripts, simulations, hand-run notebooks) as well
as the code they do. It shows that the data can be FOUND; only running the code shows that it runs.

Usage:  python RunFigures/check_code_data.py [-v] [paths ...]
    -v, --verbose   list every lookup, not only the problems
    paths           limit the scan to these files or folders (default: the whole repository)

Exit code 1 if a lookup has an unknown key, a required file is missing, or a hard-coded data path is found.
Files listed under NOT YET LOCATED in the config are reported but do not fail the test.
"""
import argparse
import ast
import importlib.util
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location('data_paths', REPO / 'Code' / 'data_paths.py')
dp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dp)  # loaded by path: importing splitflow would import torch, gurobi, ...

LOOKUPS = {'data_path', 'data_dir', 'result_path', 'unifying_results_path', 'trajectory_for_result'}
SKIP = {'Code/data_paths.py', 'Code/unifying_results.py', 'RunFigures/check_code_data.py'}  # the lookup code itself

# A string in the code that looks like a hard-coded data location.
DATA_FOLDERS = sorted(p.name for p in dp.DATA_DIR.iterdir() if p.is_dir()) if dp.DATA_DIR.is_dir() else []
HARD_CODED = re.compile(
    r'(^|/)Data/|/Data$'                               # .../Data/...
    r'|^/(media|home|Users|mnt|Volumes)/'              # another machine or an external drive
    + (r'|(^|/)(%s)(/|$)' % '|'.join(map(re.escape, DATA_FOLDERS)) if DATA_FOLDERS else ''))
NOT_DATA = re.compile(r'site-packages|^/Users/[^/]+/\.|\n')  # python paths, dot-folders, multi-line text


# ---------------------------------------------------------------------------------------------------
# Reading the code
# ---------------------------------------------------------------------------------------------------
def tracked_files(paths):
    out = subprocess.check_output(['git', 'ls-files', '-co', '--exclude-standard', '--', '*.py', '*.ipynb'],
                                  cwd=REPO).decode().split('\n')
    files = sorted(f for f in out if f and f not in SKIP and (REPO / f).exists())
    if paths:
        wanted = [str(Path(p).resolve().relative_to(REPO)) for p in paths]
        files = [f for f in files if any(f == w or f.startswith(w.rstrip('/') + '/') for w in wanted)]
    return files


def code_blocks(rel):
    """[(label, source)]: the whole file for a script, one block per code cell for a notebook."""
    path = REPO / rel
    if path.suffix == '.py':
        return [('', path.read_text())]
    cells = json.loads(path.read_text()).get('cells', [])
    blocks = []
    for i, cell in enumerate(cells):
        if cell.get('cell_type') != 'code':
            continue
        src = cell['source'] if isinstance(cell['source'], str) else ''.join(cell['source'])
        blocks.append(('cell %d' % i, src))
    return blocks


def _pattern(node):
    """
    A regex for the strings an expression can produce: literal text where it is known, '.*' elsewhere.
    Returns (regex, is_literal). Handles constants, f-strings, 'a' + b, 'a %s' % b and str(...).
    """
    if isinstance(node, ast.Constant):
        return re.escape(str(node.value)), True
    if isinstance(node, ast.JoinedStr):
        parts = [_pattern(v) if isinstance(v, ast.Constant) else ('.*', False) for v in node.values]
        return ''.join(p for p, _ in parts), all(lit for _, lit in parts)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        (a, la), (b, lb) = _pattern(node.left), _pattern(node.right)
        return a + b, la and lb
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod) and isinstance(node.left, ast.Constant) \
            and isinstance(node.left.value, str):
        text = re.split(r'%[-#0 +]*\d*(?:\.\d+)?[sdifr]', node.left.value)
        return '.*'.join(re.escape(t) for t in text), len(text) == 1
    return '.*', False


KEY_LIKE = re.compile(r'[a-z0-9_]+(\.[A-Za-z0-9_]+)+\Z')
RESULT_LIKE = re.compile(r'translation(True|False|\.\*)?\\\.parquet\Z')


def _namespaces():
    return {k.split('.')[0] for k in _all_keys()}


def _text_expressions(tree, skip):
    """(node, regex, is_literal) for every outermost string expression in the code (constants, f-strings, a + b)."""
    out = []

    def visit(node):
        if id(node) in skip:
            return
        is_text = (isinstance(node, ast.Constant) and isinstance(node.value, str)) or isinstance(node, ast.JoinedStr) \
            or (isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Mod)))
        if is_text:
            pattern, literal = _pattern(node)
            if pattern.replace('.*', ''):
                out.append((node, pattern, literal))
                return
        for child in ast.iter_child_nodes(node):
            visit(child)

    visit(tree)
    return out


def _docstrings(tree):
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.body \
                and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant):
            out.add(id(node.body[0].value))
    return out


def scan(rel):
    """(lookups, hard_coded, errors) for one file. A lookup is a dict: func, pattern, literal, kwargs, write, where."""
    lookups, hard, errors = [], [], []
    defined = set()  # lookup functions imported or defined so far (notebooks run cell by cell)
    for label, src in code_blocks(rel):
        try:
            tree = ast.parse(src)
        except SyntaxError:
            try:  # IPython magics and shell escapes are not Python
                tree = ast.parse('\n'.join('pass' if re.match(r'\s*[%!]', line) else line for line in src.split('\n')))
            except SyntaxError as err:
                errors.append('%s could not be parsed (%s)' % (label or 'file', err.msg))
                continue
        where = lambda node: ('%s line %d' % (label, node.lineno)).strip()  # noqa: E731
        in_lookup = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                defined |= {a.asname or a.name for a in node.names}
                if any(a.name == '*' for a in node.names):
                    defined |= LOOKUPS
            elif isinstance(node, ast.FunctionDef):
                defined.add(node.name)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, 'attr', None)
            if name not in LOOKUPS or not node.args:
                continue
            if isinstance(node.func, ast.Name) and name not in defined:
                hard.append('NOT IMPORTED YET: %s: %s() is used before it is imported' % (where(node), name))
            for sub in ast.walk(node):
                in_lookup.add(id(sub))
            pattern, literal = _pattern(node.args[0])
            kwargs = {k.arg: (k.value.value if isinstance(k.value, ast.Constant) else Ellipsis)
                      for k in node.keywords if k.arg}
            write = kwargs.pop('must_exist', True) is False
            lookups.append(dict(func=name, pattern=pattern, literal=literal, kwargs=kwargs, write=write,
                                where=where(node)))
        skip = _docstrings(tree) | in_lookup
        # Keys and results file names that are kept in a table or a variable and looked up later
        # (DATASETS = {'Laminar': 'fly.laminar_c1xwt'}; fname = 'laminar_data_flash_translation' + ...).
        for node, pattern, literal in _text_expressions(tree, skip):
            text = re.sub(r'\\(.)', r'\1', pattern)
            if literal and KEY_LIKE.match(text) and text.split('.')[0] in _namespaces():
                lookups.append(dict(func='data_path', pattern=pattern, literal=True, kwargs={}, write=False,
                                    where=where(node), implicit=True))
            elif RESULT_LIKE.search(pattern):
                lookups.append(dict(func='result_path', pattern=pattern, literal=literal, kwargs={}, write=False,
                                    where=where(node), implicit=True))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in skip \
                    and HARD_CODED.search(node.value) and not NOT_DATA.search(node.value):
                hard.append('HARD-CODED PATH: %s: %r' % (where(node), node.value[:110]))
    return lookups, hard, errors


# ---------------------------------------------------------------------------------------------------
# Checking the lookups against the config and the disk
# ---------------------------------------------------------------------------------------------------
def _all_keys():
    cfg = dp.config()
    return [k for section in ('files', 'directories', 'not_yet_located') for k in cfg[section]]


def check(lookup, rows_by_key, result_names):
    """[(status, text)] for one lookup. status: ok | located_later | optional | dynamic | error."""
    func, pattern = lookup['func'], lookup['pattern']
    regex = re.compile(pattern + r'\Z')
    shown = pattern if not lookup['literal'] else re.sub(r'\\(.)', r'\1', pattern)

    if func in ('data_path', 'data_dir'):
        keys = [k for k in _all_keys() if regex.match(k)]
        if not lookup['literal'] and pattern.replace('.*', '') == '':
            return [('dynamic', '%s(<variable>): key decided at run time' % func)]
        if not keys:
            return [('error', "%s('%s'): no such key in %s" % ('key' if lookup.get('implicit') else func, shown,
                                                               dp.config_file().name))]
        out = []
        for key in keys:
            rows = [r for r in rows_by_key[key]
                    if all(v is Ellipsis or str(r['fields'].get(k)) == str(v) for k, v in lookup['kwargs'].items())]
            unknown = [k for k in lookup['kwargs'] if k not in dp.placeholders(dp.entry(key)[1]['path'])]
            if unknown:
                out.append(('error', "%s('%s'): the path has no {%s} placeholder" % (func, key, '}, {'.join(unknown))))
            elif not rows:
                given = ', '.join('%s=%r' % kv for kv in lookup['kwargs'].items() if kv[1] is not Ellipsis)
                out.append(('error', "%s('%s', %s): not one of the listed `values`" % (func, key, given)))
            for r in rows:
                text = '%s -> %s' % (dp._label(r), dp._rel(r['path']))
                if r['status'] in ('ok', 'located') or lookup['write']:
                    out.append(('ok', text + ('  (written here)' if lookup['write'] else '')))
                elif r['status'] == 'not_located':
                    out.append(('located_later', text))
                elif r['status'] == 'not_generated':
                    out.append(('optional', text))
                else:
                    out.append(('error', '%s: %s' % (text, r['detail'])))
        return out

    # result_path / trajectory_for_result: looked up by results file name
    names = [n for n in result_names if regex.match(n)]
    if lookup['literal']:
        names = [re.sub(r'\\(.)', r'\1', pattern)]
    elif pattern.replace('.*', '') == '':
        return [('dynamic', '%s(<variable>): file name decided at run time' % func)]
    if not names:
        if lookup['write']:
            return [('ok', "%s('%s')  (written here; not read by any figure)" % (func, shown))]
        if lookup.get('implicit') and dp.subdir_for(shown.replace('.*', 'True')) == dp._deprecated_dir():
            return []  # a results file name no figure uses (kept in a table; result_path would refuse it)
        return [('error', "%s('%s'): matches no file under unifying_results.current" % (func, shown))]
    out = []
    for name in names:
        if func == 'trajectory_for_result':
            try:
                key = dp.trajectory_key_for_result(name)
            except ValueError as err:
                out.append(('error', str(err)))
                continue
            bad = [r for r in rows_by_key[key] if r['status'] not in ('ok', 'located')]
            out.append(('error' if bad else 'ok', '%s -> %s' % (name, key)))
            continue
        subdir = dp.subdir_for(name)
        path = dp.results_dir() / subdir / name
        text = '%s -> %s' % (name, dp._rel(path))
        if lookup['write']:
            out.append(('ok', text + '  (written here)'))
        elif subdir == dp._deprecated_dir():
            out.append(('error', '%s: deprecated (no translation term, or matches no layout rule)' % name))
        elif not path.is_file():
            out.append(('error', '%s: file not found; make it with: %s' % (text, dp.how_to_make(name))))
        else:
            out.append(('ok', text))
    return out


def run_by_figure_scripts():
    """Scripts and notebooks the RunFigures/*.sh scripts execute (paths relative to the repository)."""
    out = set()
    for sh in sorted((REPO / 'RunFigures').glob('*.sh')):
        variables = {}
        for line in sh.read_text().split('\n'):
            line = line.strip()
            m = re.match(r'(\w+)=("?)([\w./-]+)\2$', line)
            if m:
                variables[m.group(1)] = m.group(3)
            m = re.match(r'run_(?:notebook|python)\s+"?(\$\{?(\w+)\}?|[\w./-]+)"?\s+([\w.-]+)', line)
            if m:
                folder = variables.get(m.group(2), '') if m.group(2) else m.group(1)
                out.add('%s/%s' % (folder.rstrip('/'), m.group(3)))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0].strip())
    parser.add_argument('paths', nargs='*')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    rows_by_key = defaultdict(list)
    for row in dp.check_entries():
        rows_by_key[row['key']].append(row)
    result_names = dp.current_results()
    in_figures = run_by_figure_scripts()

    files = tracked_files(args.paths)
    report = {}  # file -> (results, hard_coded, errors)
    for rel in files:
        lookups, hard, errors = scan(rel)
        results = [(lookup, check(lookup, rows_by_key, result_names)) for lookup in lookups]
        if results or hard or errors:
            report[rel] = (results, hard, errors)

    totals = defaultdict(int)
    n_hard = n_parse = 0
    print('config file: %s' % dp.config_file())
    print('scanned %d scripts and notebooks; %d of them look up data\n' % (len(files), len(report)))
    groups = (('RUN BY THE RunFigures SCRIPTS', lambda f: f in in_figures),
              ('NOT RUN BY THE RunFigures SCRIPTS (helper modules, fit and simulation scripts, hand-run notebooks)',
               lambda f: f not in in_figures))
    for title, member in groups:
        print('=' * 100)
        print(title)
        print('=' * 100)
        for rel in sorted(f for f in report if member(f)):
            results, hard, errors = report[rel]
            counts = defaultdict(int)
            lines = []
            for lookup, outcome in results:
                for status, text in outcome:
                    counts[status] += 1
                    totals[status] += 1
                    if status not in ('ok', 'dynamic') or args.verbose:
                        tag = {'ok': 'ok', 'error': 'PROBLEM', 'located_later': 'not yet located',
                               'optional': 'not generated (optional)', 'dynamic': 'dynamic'}[status]
                        lines.append('      %-16s %s  [%s]' % (tag + ':', text, lookup['where']))
            for h in hard:
                lines.append('      ' + h)
            for e in errors:
                lines.append('      %-16s %s' % ('NOT CHECKED:', e))
            n_hard += len(hard)
            n_parse += len(errors)
            summary = ', '.join('%d %s' % (counts[s], label) for s, label in
                                (('ok', 'found'), ('error', 'PROBLEM'), ('located_later', 'not yet located'),
                                 ('optional', 'optional, not generated'), ('dynamic', 'dynamic')) if counts[s])
            flag = 'FAIL' if counts['error'] or hard else ('warn' if counts['located_later'] or errors else ' ok ')
            print('[%s] %s: %s%s' % (flag, rel, summary or 'no lookups',
                                     ', %d hard-coded path or import problem(s)' % len(hard) if hard else ''))
            for line in lines:
                print(line)
        print()

    print('lookups resolved to existing data: %d' % totals['ok'])
    print('lookups of files listed as NOT YET LOCATED: %d' % totals['located_later'])
    if totals['optional']:
        print('lookups of optional, not yet generated files: %d' % totals['optional'])
    if totals['dynamic']:
        print('lookups through a variable (the keys and file names they can take are checked where they are written): %d'
              % totals['dynamic'])
    if n_parse:
        print('code blocks that could not be parsed (not checked): %d' % n_parse)
    print('PROBLEMS: %d unknown or missing, %d hard-coded data path(s) or lookup(s) used before their import'
          % (totals['error'], n_hard))
    print()
    if totals['error'] or n_hard:
        print('FAILED')
        return 1
    print('OK: every data lookup in the code resolves to data that is in place'
          + (' (apart from the files not yet located).' if totals['located_later'] else '.'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
