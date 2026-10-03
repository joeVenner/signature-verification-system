"""Isolate handwritten signature ink from printed context on the page.

`normalize_signature` crops to the extent of *all* ink, so ruled lines, form
rules and printed captions ("PAY TO THE ORDER OF") inflate the crop and leak
into every descriptor. This module removes two kinds of non-signature ink from
the darkness map before cropping:

* **Long horizontal rules** — a 1-px-tall grayscale opening keeps only runs
  that are perfectly horizontal over half the image width. Some hand-written
  underlines survive it too, so a run is removed only if it is fainter than
  ink and not the edge of a stroke, or ink-dark and standing alone.
* **Printed text lines** — groups of ink components that look like a line of
  type: many glyphs of similar height sitting on one baseline, much wider than
  tall. Distance and size rules cannot be used instead: CEDAR signatures have
  detached parts (initials, surnames) up to 4x the main part's height away and
  of equal mass (see benchmark/experiments.md EXP-016).

Everything is deterministic: fixed kernels, OpenCV raster-order labels and no
iteration over unordered collections. When nothing is detected the darkness
map is returned unchanged (the same array object); 659/660 clean dev images
are bit-identical to the pre-isolation pipeline.
"""

from __future__ import annotations

import cv2
import numpy as np

# Rule length as a fraction of image width. EXP-016 (660 dev images): with
# the two filters below, rules were removed in 660/660 ruled_lines and
# distractor_print images and in 1/660 clean images (a scanner border line on
# the top edge of a forgery scan).
RULE_LENGTH_FRAC = 0.5
RULE_MIN_LENGTH_PX = 60       # never treat short dashes as rules on tiny inputs
# An ink-dark run is a printed rule only if it makes up this share of the ink
# components it touches, i.e. it stands alone instead of continuing a stroke.
RULE_MIN_PURITY = 0.6
# A faint (sub-ink) run is a ruled line only if at most this share of it lies
# within DROP_HALO_PX rows of ink; more means it is the edge of an ink stroke.
RULE_MAX_INK_CONTACT = 0.5

# Glyph grouping radius as a fraction of the image diagonal: wide enough to
# join the letters and words of a printed line into one group (word spaces
# are ~0.6x the glyph height) on every canvas size in the benchmark.
GROUP_RADIUS_FRAC = 0.02
GROUP_MIN_RADIUS_PX = 2
MIN_GLYPH_AREA_PX = 4         # smaller components are specks, not glyphs

# A group is a printed text line if it has at least TEXT_MIN_GLYPHS glyphs,
# every glyph sits on one line (group height <= TEXT_MAX_LINE_HEIGHT x the
# median glyph height) and it is at least TEXT_MIN_ASPECT times wider than
# tall. EXP-016: flagged in 0/660 clean dev images and 660/660
# distractor_print dev images.
TEXT_MIN_GLYPHS = 6
TEXT_MAX_LINE_HEIGHT = 1.5
TEXT_MIN_ASPECT = 6.0
# Dropped groups are cleared over a slightly dilated footprint so the
# anti-aliased halo below the ink threshold does not stay in the crop.
DROP_HALO_PX = 2


def remove_horizontal_rules(darkness: np.ndarray, ink_level: float) -> np.ndarray:
    """Subtract long, perfectly horizontal runs that are printed rules.

    A run is a rule if it is fainter than ink everywhere (ruled paper: the
    signature crosses it but never merges with it), or if it is ink-dark and
    forms its own ink component(s) (a form rule away from the signature).
    Ink-dark runs fused with handwriting are kept: CEDAR underline flourishes
    are perfectly straight for 0.4-0.9 of the image width (EXP-016).

    Args:
        darkness: float ink-darkness map in [0, 1], 0 = paper.
        ink_level: darkness above which a pixel counts as ink.

    Returns:
        The input array itself if no rule is found, else a cleaned copy.
    """
    # Odd length keeps the kernel centred, so a full-width rule is recovered
    # up to both image edges.
    length = max(RULE_MIN_LENGTH_PX, int(round(RULE_LENGTH_FRAC * darkness.shape[1]))) | 1
    if length > darkness.shape[1]:
        return darkness
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (length, 1))
    # Zero padding: OpenCV's default erosion border is +inf, which would let a
    # stroke touching the image edge pass with only half the rule length.
    runs = cv2.morphologyEx(darkness, cv2.MORPH_OPEN, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    if not np.any(runs > 0):
        return darkness

    n_runs, run_labels = cv2.connectedComponents((runs > 0).astype(np.uint8), connectivity=8)
    ink = darkness > ink_level
    n_ink, ink_labels, ink_stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    near_ink = cv2.dilate(ink.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2 * DROP_HALO_PX + 1))) > 0
    rules = np.zeros_like(darkness)
    for r in range(1, n_runs):
        run = run_labels == r
        if float(runs[run].max()) <= ink_level:
            # Faint run: a ruled line only touches ink where strokes cross it;
            # the anti-aliased edge of a straight ink stroke runs along it.
            if float(near_ink[run].mean()) > RULE_MAX_INK_CONTACT:
                continue
        else:
            run_ink = run & ink
            touched = np.unique(ink_labels[run_ink])
            if np.count_nonzero(run_ink) < RULE_MIN_PURITY * ink_stats[touched, cv2.CC_STAT_AREA].sum():
                continue
        rules[run] = runs[run]
    if not np.any(rules > 0):
        return darkness
    return np.clip(darkness - rules, 0.0, 1.0)


def _text_line_groups(mask: np.ndarray) -> np.ndarray:
    """Label map of the ink groups that look like a printed text line.

    Returns:
        int32 array, same shape as `mask`, with the group label of every pixel
        belonging to a text-like group and 0 elsewhere.
    """
    radius = max(GROUP_MIN_RADIUS_PX, int(round(GROUP_RADIUS_FRAC * float(np.hypot(*mask.shape)))))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    mask_u8 = mask.astype(np.uint8)
    n_groups, groups = cv2.connectedComponents(cv2.dilate(mask_u8, kernel), connectivity=8)
    if n_groups <= 2:              # background + a single group: nothing to separate
        return np.zeros_like(groups)

    n_glyphs, glyphs, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    # Every glyph lies inside exactly one dilated group.
    group_of_glyph = np.zeros(n_glyphs, np.int32)
    group_of_glyph[glyphs[mask]] = groups[mask]

    text_labels = []
    for g in range(1, n_groups):
        members = np.flatnonzero((group_of_glyph == g) & (stats[:, cv2.CC_STAT_AREA] >= MIN_GLYPH_AREA_PX))
        members = members[members > 0]
        if len(members) < TEXT_MIN_GLYPHS:
            continue
        s = stats[members]
        top = int(s[:, cv2.CC_STAT_TOP].min())
        bottom = int((s[:, cv2.CC_STAT_TOP] + s[:, cv2.CC_STAT_HEIGHT]).max())
        left = int(s[:, cv2.CC_STAT_LEFT].min())
        right = int((s[:, cv2.CC_STAT_LEFT] + s[:, cv2.CC_STAT_WIDTH]).max())
        height, width = bottom - top, right - left
        glyph_height = float(np.median(s[:, cv2.CC_STAT_HEIGHT]))
        if height <= TEXT_MAX_LINE_HEIGHT * glyph_height and width >= TEXT_MIN_ASPECT * height:
            text_labels.append(g)
    if not text_labels or len(text_labels) == n_groups - 1:
        # Never drop everything: an all-text input is better cropped as-is.
        return np.zeros_like(groups)
    return np.where(np.isin(groups, text_labels) & mask, groups, 0).astype(np.int32)


def isolate_signature_ink(darkness: np.ndarray, ink_level: float) -> np.ndarray:
    """Remove printed rules and text lines from an ink-darkness map.

    Args:
        darkness: float ink-darkness map in [0, 1], 0 = paper.
        ink_level: darkness above which a pixel counts as ink.

    Returns:
        The input array itself when nothing is removed (or when removal would
        leave no ink at all, e.g. a single straight stroke), else a cleaned copy.
    """
    cleaned = remove_horizontal_rules(darkness, ink_level)
    if not np.any(cleaned > ink_level):
        return darkness
    text = _text_line_groups(cleaned > ink_level)
    if not np.any(text):
        return cleaned
    halo = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * DROP_HALO_PX + 1, 2 * DROP_HALO_PX + 1))
    footprint = cv2.dilate((text > 0).astype(np.uint8), halo) > 0
    out = cleaned.copy()
    out[footprint] = 0.0
    return out
