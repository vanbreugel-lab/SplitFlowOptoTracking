"""
Where the data lives. Every script and notebook asks this module for its data files; the names and
locations themselves are listed in Data/data_paths.yaml, so data can be renamed or moved by editing
that file only.

    from splitflow.data_paths import data_path, data_dir
    df = pd.read_hdf(data_path('fly.laminar_c1xwt'))                 # a file
    df = pd.read_csv(data_path('wind.laminar', speed=40))           # a file whose path has {placeholders}
    folder = data_dir('cfd.xz_slice')                               # a folder
    out = data_path('shuffled.flies', wind='laminar', must_exist=False)   # a file about to be written

Unifying-algorithm fit results are looked up by file name (see splitflow.unifying_results):

    from splitflow.unifying_results import result_path
    fits = pd.read_parquet(result_path('laminar_data_flash_translationTrue.parquet'))
    trajectories = trajectory_for_result('laminar_data_flash_translationTrue.parquet')

Environment variables (optional):
    SPLITFLOW_DATA_DIR    the data directory (default: <repository>/Data)
    SPLITFLOW_DATA_PATHS  the config file (default: data_paths.yaml in the data directory)

Run as a script to check that the data is in place (this is what Data/check_data.sh does):

    python Code/data_paths.py check [--deep] [--unlisted]

This module needs only the standard library and PyYAML, and does not import the rest of splitflow.
"""
import itertools
import os
import re
import string
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get('SPLITFLOW_DATA_DIR') or REPO_ROOT / 'Data').resolve()

_SECTIONS = ('files', 'directories', 'not_yet_located')
_config_cache = {}


def config_file():
    """The config file: $SPLITFLOW_DATA_PATHS, else data_paths.yaml in the data directory (else in <repo>/Data)."""
    if os.environ.get('SPLITFLOW_DATA_PATHS'):
        return Path(os.environ['SPLITFLOW_DATA_PATHS']).resolve()
    path = DATA_DIR / 'data_paths.yaml'
    return path if path.exists() else REPO_ROOT / 'Data' / 'data_paths.yaml'


def config():
    """The parsed config file (read once)."""
    path = config_file()
    if path not in _config_cache:
        with open(path) as f:
            cfg = yaml.safe_load(f) or {}
        for section in _SECTIONS:
            cfg[section] = cfg.get(section) or {}
        _config_cache[path] = cfg
    return _config_cache[path]


# ---------------------------------------------------------------------------------------------------
# Entries
# ---------------------------------------------------------------------------------------------------
def entry(key):
    """(section, entry as a dict) for a key; entries written as a bare path become {'path': ...}."""
    cfg = config()
    for section in _SECTIONS:
        if key in cfg[section]:
            e = cfg[section][key]
            return section, ({'path': e} if isinstance(e, str) else dict(e))
    raise KeyError("no data entry '%s' in %s" % (key, config_file()))


def placeholders(template):
    """Names of the {placeholders} in a path template."""
    return [name for _, name, _, _ in string.Formatter().parse(template) if name]


def _resolve(e, fields):
    template = e['path']
    missing = [name for name in placeholders(template) if name not in fields]
    if missing:
        raise TypeError("path '%s' needs %s" % (template, ', '.join('%s=...' % m for m in missing)))
    path = Path(template.format(**fields))
    if path.is_absolute():
        return path
    return (REPO_ROOT if e.get('root') == 'repo' else DATA_DIR) / path


def _is_directory(section, e):
    return section == 'directories' or e.get('kind') == 'directory'


def _missing_message(key, section, e, path):
    msg = "%s not found (data entry '%s' in %s)." % (path, key, config_file())
    if section == 'not_yet_located':
        msg += ' It is listed under NOT YET LOCATED: this file has not been found on this machine.'
        if e.get('note'):
            msg += ' Note: %s.' % e['note']
    elif e.get('made_by'):
        msg += ' Make it with: %s' % e['made_by']
    return msg


def data_path(key, must_exist=True, **fields):
    """
    Full path of a data file, given its key in Data/data_paths.yaml.

    fields fill the {placeholders} of the entry's path. must_exist=False returns the path a script
    should write to, creating its folder.
    """
    section, e = entry(key)
    path = _resolve(e, fields)
    if not must_exist:
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    if not path.exists():
        raise FileNotFoundError(_missing_message(key, section, e, path))
    return path


def data_dir(key, must_exist=True, **fields):
    """Full path of a data folder, given its key in Data/data_paths.yaml. must_exist=False creates it."""
    section, e = entry(key)
    path = _resolve(e, fields)
    if not must_exist:
        path.mkdir(parents=True, exist_ok=True)
        return path
    if not path.is_dir():
        raise FileNotFoundError(_missing_message(key, section, e, path))
    return path


def expand(e):
    """Every (fields, path) an entry stands for: one per combination of its `values`."""
    names = placeholders(e['path'])
    values = e.get('values') or {}
    unknown = [n for n in names if n not in values]
    if unknown:
        raise KeyError("path '%s' has no `values` for %s" % (e['path'], ', '.join(unknown)))
    for combo in itertools.product(*(values[n] for n in names)):
        fields = dict(zip(names, combo))
        yield fields, _resolve(e, fields)


# ---------------------------------------------------------------------------------------------------
# Unifying-algorithm fit results (looked up by file name)
# ---------------------------------------------------------------------------------------------------
def _results_config():
    return config().get('unifying_results') or {}


def results_dir():
    """The results directory: the first of `unifying_results.dir_names` that exists in the data directory."""
    names = _results_config().get('dir_names') or ['unifying_algo_results']
    for name in names:
        if (DATA_DIR / name).is_dir():
            return DATA_DIR / name
    return DATA_DIR / names[0]


def _deprecated_dir():
    return _results_config().get('deprecated_dir', 'deprecated')


def subdir_for(filename):
    """Subdirectory of the results directory that holds `filename`."""
    name = Path(filename).name
    if 'translationFalse' in name:
        return _deprecated_dir()  # fits without the translation term are no longer used
    for pattern, subdir in _results_config().get('layout') or []:
        m = re.match(pattern, name)
        if m:
            return m.expand(subdir)
    return _deprecated_dir()  # datasets no figure uses


def how_to_make(filename):
    """The command that makes a results file (None if unknown)."""
    name = Path(filename).name
    return next((cmd for pattern, cmd in _results_config().get('how_to_make') or [] if re.match(pattern, name)), None)


def result_path(filename, must_exist=True):
    """
    Full path of a results file, given its file name (any directory part is ignored).

    must_exist=False returns the path a fit script should write to, creating the subdirectory.
    """
    name = Path(filename).name
    subdir = subdir_for(name)
    path = results_dir() / subdir / name
    if not must_exist:
        if subdir == _deprecated_dir():
            # a fit script was run without --include-translation, or for a dataset no figure uses
            print('NOTE: %s is not used by any figure; writing it to %s/' % (name, subdir))
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    if subdir == _deprecated_dir():
        raise FileNotFoundError('%s is deprecated (no translation term, or not used by any figure); '
                                'it is kept in %s but is not a current result.' % (name, path.parent))
    if not path.exists():
        raise FileNotFoundError('%s not found. Make it with:  python %s' % (path, how_to_make(name)))
    return path


def trajectory_key_for_result(filename):
    """Key of the trajectory file a results file was fit to (`unifying_results.trajectories`)."""
    name = Path(filename).name
    for pattern, key in _results_config().get('trajectories') or []:
        if re.search(pattern, name):
            return key
    raise ValueError('No trajectory file known for ' + name)


def trajectory_for_result(filename, must_exist=True):
    """Full path of the trajectory file a results file was fit to."""
    return data_path(trajectory_key_for_result(filename), must_exist=must_exist)


def current_results():
    """File names of every result the figures and scripts read (`unifying_results.current`, expanded)."""
    rc = _results_config()
    values = rc.get('values') or {}
    names = []
    for template in rc.get('current') or []:
        fields = placeholders(template)
        for combo in itertools.product(*(values[f] for f in fields)):
            names.append(template.format(**dict(zip(fields, combo))))
    return names


# ---------------------------------------------------------------------------------------------------
# The data check
# ---------------------------------------------------------------------------------------------------
def _rel(path):
    for root, prefix in ((DATA_DIR, ''), (REPO_ROOT, '<repo>/')):
        try:
            return prefix + str(Path(path).relative_to(root))
        except ValueError:
            pass
    return str(path)


def _directory_problem(path, e):
    """None if the folder exists and holds what the entry says it should; otherwise what is wrong."""
    if not path.is_dir():
        return 'folder not found'
    patterns = e.get('contains') or []
    for pattern in patterns:
        n = sum(1 for _ in path.glob(pattern))
        need = e.get('min_files', 1) if len(patterns) == 1 else 1
        if n < need:
            return 'has %d file(s) matching %s, expected at least %d' % (n, pattern, need)
    return None


def _open_problem(path):
    """--deep: None if the file's header can be read; otherwise the error."""
    try:
        suffix = path.suffix.lower()
        if suffix == '.parquet':
            import pyarrow.parquet as pq
            pq.read_schema(path)
        elif suffix in ('.hdf', '.h5'):
            import h5py
            with h5py.File(path, 'r') as f:
                if not len(f.keys()):
                    return 'no datasets in the file'
        elif suffix in ('.csv', '.dat'):
            with open(path, 'rb') as f:
                if not f.readline().strip():
                    return 'first line is empty'
        elif path.stat().st_size == 0:
            return 'empty file'
    except Exception as err:  # noqa: BLE001 - any failure to open is the finding
        return '%s: %s' % (type(err).__name__, err)
    return None


def check_entries(deep=False):
    """
    Check every entry of the config against the disk.

    Returns a list of dicts: key, section, path, status ('ok' | 'missing' | 'unreadable' | 'not_generated' |
    'not_located' | 'located'), detail, entry.
    """
    cfg = config()
    rows = []
    for section in _SECTIONS:
        for key in cfg[section]:
            _, e = entry(key)
            for fields, path in expand(e):
                if _is_directory(section, e):
                    problem = _directory_problem(path, e)
                else:
                    problem = None if path.is_file() else 'file not found'
                status, detail = 'ok', ''
                if problem:
                    status, detail = 'missing', problem
                    if section == 'not_yet_located':
                        status = 'not_located'
                    elif e.get('optional'):
                        status = 'not_generated'
                elif section == 'not_yet_located':
                    status = 'located'
                elif deep and path.is_file():
                    problem = _open_problem(path)
                    if problem:
                        status, detail = 'unreadable', problem
                rows.append(dict(key=key, fields=fields, section=section, path=path, status=status, detail=detail,
                                 entry=e))
    for pattern, key in _results_config().get('trajectories') or []:
        if not any(key in cfg[section] for section in _SECTIONS):
            rows.append(dict(key='unifying_results', fields={}, section='unifying_results', path=config_file(),
                             status='missing', entry={},
                             detail="unifying_results.trajectories refers to '%s', which is not a key" % key))
    for name in current_results():
        subdir = subdir_for(name)
        path = results_dir() / subdir / name
        status, detail = 'ok', ''
        if subdir == _deprecated_dir():
            status, detail = 'missing', 'file name matches no `layout` rule'
        elif not path.is_file():
            status, detail = 'missing', 'file not found'
        elif deep:
            problem = _open_problem(path)
            if problem:
                status, detail = 'unreadable', problem
        rows.append(dict(key='unifying_results', fields={}, section='unifying_results', path=path, status=status,
                         detail=detail, entry={'made_by': how_to_make(name)}))
    return rows


def unlisted():
    """
    Files and folders in the data directory that no config entry covers (candidates for cleanup).

    A folder is reported as a whole when nothing inside it is covered.
    """
    covered_files, covered_dirs = set(), set()
    for row in check_entries():
        path = row['path']
        if DATA_DIR not in path.parents:
            continue
        if row['section'] == 'directories' or row['entry'].get('kind') == 'directory':
            covered_dirs.add(path)
        else:
            covered_files.add(path)
    if results_dir().is_dir():
        covered_files.add(results_dir() / 'README.md')
    keep = {config_file(), DATA_DIR / 'check_data.sh', DATA_DIR / 'download_from_dryad_and_unzip_here.txt'}
    for pattern in config().get('expected_unlisted') or []:
        keep |= set(DATA_DIR.glob(pattern))

    def is_covered(path):
        return path in covered_files or path in keep or any(d == path or d in path.parents for d in covered_dirs)

    def has_covered(folder):
        return any(folder in p.parents for p in covered_files | covered_dirs | keep)

    out = []

    def walk(folder):
        for child in sorted(folder.iterdir()):
            if child.name == '.DS_Store' or is_covered(child):
                continue
            if child.is_dir():
                if child in covered_dirs:
                    continue
                if has_covered(child):
                    walk(child)
                else:
                    out.append(child)
            else:
                out.append(child)

    if DATA_DIR.is_dir():
        walk(DATA_DIR)
    return out


def _size(path):
    if path.is_dir():
        n = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
    else:
        n = path.stat().st_size
    for unit in ('B', 'kB', 'MB', 'GB'):
        if n < 1024 or unit == 'GB':
            return '%.0f %s' % (n, unit) if unit in ('B', 'kB') else '%.1f %s' % (n, unit)
        n /= 1024


def _label(row):
    fields = ', '.join('%s=%s' % kv for kv in row['fields'].items())
    return row['key'] + (' (%s)' % fields if fields else '')


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description='Check that the data listed in Data/data_paths.yaml is in place.')
    parser.add_argument('command', choices=['check'])
    parser.add_argument('--deep', action='store_true', help="also open each file's header (parquet schema, HDF keys)")
    parser.add_argument('--unlisted', action='store_true',
                        help='also list files in the data directory that no entry covers')
    args = parser.parse_args(argv)

    print('data directory: %s' % DATA_DIR)
    print('config file:    %s' % config_file())
    rows = check_entries(deep=args.deep)
    by = lambda status: [r for r in rows if r['status'] == status]  # noqa: E731
    bad = by('missing') + by('unreadable')

    print('\n%d of %d required files and folders are in place%s.'
          % (len(by('ok')), len(by('ok')) + len(bad), ' and readable' if args.deep else ''))
    if bad:
        print('\nMISSING (%d) - required by the code' % len(bad))
        for r in bad:
            print('  %-42s %s' % (_label(r), _rel(r['path'])))
            print('  %-42s   %s%s' % ('', r['detail'],
                                      '; make it with: %s' % r['entry']['made_by'] if r['entry'].get('made_by') else ''))
    if by('not_generated'):
        print('\nNOT GENERATED (%d) - optional outputs of regeneration scripts; no figure reads them'
              % len(by('not_generated')))
        for r in by('not_generated'):
            print('  %-42s %s' % (_label(r), _rel(r['path'])))
            print('  %-42s   make it with: %s' % ('', r['entry'].get('made_by')))
    not_located = by('not_located')
    if not_located or by('located'):
        print('\n' + '=' * 100)
        print('NOT YET LOCATED (%d) - raw inputs of hand-run notebooks and simulation scripts; no figure script needs them'
              % len(not_located))
        print('=' * 100)
        last_key = None
        for r in not_located:
            print('  %-42s %s' % (_label(r), _rel(r['path'])))
            if r['key'] != last_key:
                for used in r['entry'].get('used_by') or []:
                    print('  %-42s   used by: %s' % ('', used))
                if r['entry'].get('note'):
                    print('  %-42s   note: %s' % ('', r['entry']['note']))
            last_key = r['key']
        for r in by('located'):
            print('  FOUND: %s is now at %s; move its entry out of not_yet_located' % (_label(r), _rel(r['path'])))
    if args.unlisted:
        extra = unlisted()
        print('\nUNLISTED (%d) - in the data directory, but no entry in the config refers to them' % len(extra))
        for path in extra:
            print('  %-70s %10s' % (_rel(path) + ('/' if path.is_dir() else ''), _size(path)))

    print()
    if bad:
        print('FAILED: %d required file(s) or folder(s) missing or unreadable.' % len(bad))
        return 1
    print('OK: all required data is in place.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
