"""
Scenario metadata shared by the shuffled / synthetic UAA supplemental figure notebooks.

Paths are relative to this folder (notebooks are run from their own directory).
"""
from pathlib import Path

import pandas as pd

FIGURE_NAME = 'supplemental_unifying_analysis_shuffled_synthetic.svg'
TRANSLATION = True

DATA_DIR = Path('../../../Data')
SHUFFLED_DIR = DATA_DIR / 'Shuffled_Flies'                # made by time_shift_flies/make_shuffled_datasets.py
SYNTHETIC_DIR = DATA_DIR / 'Synthetic_Flies_for_Unifying'  # made by uaa_on_synthetic_trajectories/extract_synthetic_trajectories.py
RESULTS_DIR = DATA_DIR / 'unifying_algo_results'          # made by analyze_shuffled.py / analyze_synthetic.py

# Ordered left -> right in the figure
SCENARIOS = ['shuffled_laminar', 'shuffled_stillair', 'levy_flight', 'random_turns', 'brownian', 'run_and_tumble']

# Panel C splits slope and MI into two axes: the high-slope / high-MI scenarios get their own y-range
HIGH_SCENARIOS = ['brownian', 'run_and_tumble']
LOW_SCENARIOS = [key for key in SCENARIOS if key not in HIGH_SCENARIOS]

TITLES = {'shuffled_laminar': 'Shuffled laminar',
          'shuffled_stillair': 'Shuffled still air',
          'brownian': 'Brownian',
          'levy_flight': 'Lévy flight',
          'random_turns': 'Random turns',
          'run_and_tumble': 'Run and tumble'}

# Short labels for categorical x-axes
SHORT_LABELS = {'shuffled_laminar': 'Shuf. laminar',
                'shuffled_stillair': 'Shuf. still air',
                'brownian': 'Brownian',
                'levy_flight': 'Lévy',
                'random_turns': 'Rand. turns',
                'run_and_tumble': 'Run & tumble'}

SCENARIO_COLOR = 'gray'  # all scenarios, same gray as the sham data in the variable-wind figure
COLORS = {key: SCENARIO_COLOR for key in SCENARIOS}

_SHUFFLED_STEM = {'shuffled_laminar': 'SHUFFLED_flies_laminar_c1xwt_preprocessed_optotrigger_trimmed.hdf',
                  'shuffled_stillair': 'SHUFFLED_flies_stillair_c1xwt_preprocessed_optotrigger_trimmed.hdf'}


def get_unifying_filename(key):
    if key in _SHUFFLED_STEM:
        return RESULTS_DIR / f'SHUFFLED_{_SHUFFLED_STEM[key]}_translation{TRANSLATION}.parquet'
    return RESULTS_DIR / f'synthetic_{key}_translation{TRANSLATION}.parquet'


def get_trajec_filename(key):
    if key in _SHUFFLED_STEM:
        return SHUFFLED_DIR / _SHUFFLED_STEM[key]
    return SYNTHETIC_DIR / f'synthetic_{key}_n200.parquet'


def load_unifying(key):
    return pd.read_parquet(get_unifying_filename(key))


def load_trajec(key):
    fname = get_trajec_filename(key)
    if fname.suffix == '.hdf':
        return pd.read_hdf(fname)
    return pd.read_parquet(fname)


def load_scenario(key):
    """(unifying_algo_data, trajectory dataframe) for one scenario."""
    return load_unifying(key), load_trajec(key)


# --------------------------------------------------------------------------------------------------
# SVG output (same as Figure 5, slope_r2_helper.save_figure)
# --------------------------------------------------------------------------------------------------
def uniquify_glyph_ids(svg_filename, layer_names):
    """
    Prefix matplotlib glyph ids (e.g. Helvetica-30, Cmr10-31) inside the given layers with the layer name, so
    layers written in separate runs can't pick up each other's glyph definitions (otherwise characters can
    render in the wrong font or as the wrong glyph). Copied from Figure 5's slope_r2_helper.
    """
    import re
    s = open(svg_filename).read()
    for layer in layer_names:
        start = re.search(r'<g [^>]*id="%s"[^>]*inkscape:groupmode="layer"' % layer, s).start()
        depth = 0
        for m in re.finditer(r'<g(?:\s[^>]*)?>|</g>', s[start:]):
            if m.group(0).endswith('/>'):
                continue  # empty self-closing group (e.g. an axis turned off), opens nothing
            depth += 1 if m.group(0) != '</g>' else -1
            if depth == 0:
                end = start + m.end()
                break
        block = s[start:end]
        glyph_ids = set(re.findall(r'<path id="([^"]+)" d=', block)) & set(re.findall(r'xlink:href="#([^"]+)"', block))
        for gid in glyph_ids:
            if gid.startswith(layer + '_'):
                continue  # already prefixed in an earlier run
            block = block.replace('id="%s"' % gid, 'id="%s_%s"' % (layer, gid))
            block = block.replace('xlink:href="#%s"' % gid, 'xlink:href="#%s_%s"' % (layer, gid))
        s = s[:start] + block + s[end:]
    open(svg_filename, 'w').write(s)


def save_layer(layout, fifi_figure_label, figure_name=FIGURE_NAME):
    """Write only this figurefirst figure into its own layer of the shared SVG."""
    layout.append_figure_to_layer(layout.figures[fifi_figure_label], fifi_figure_label, cleartarget=True)
    layout.write_svg(figure_name)
    uniquify_glyph_ids(figure_name, [fifi_figure_label])
