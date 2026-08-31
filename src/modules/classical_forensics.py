"""
classical_forensics.py
======================
Signal-level ("classical") forensic feature extraction for AI-generated-image
detection, designed to be fused with LoRA-adapted SigLIP 2 features.

Design contract
---------------
    extractor = ClassicalFeatureExtractor()
    feats = extractor(img_uint8_rgb)          # HxWx3 uint8, NATIVE resolution

    feats["spectral"] -> (P, D_spec)   float32   per-patch
    feats["dct"]      -> (P, D_dct)    float32   per-patch
    feats["residual"] -> (P, D_res)    float32   per-patch
    feats["wavelet"]  -> (P, D_wav)    float32   per-patch
    feats["color"]    -> (P, D_col)    float32   per-patch
    feats["global"]   -> (D_glob,)     float32   whole image (degradation descriptor)
    feats["patch_meta"] -> (P, 4)      float32   [is_rich, y/H, x/W, log(ps)]

Each row is one token for the downstream attention pooler. `extractor.dims()`
returns the dimensionality of every family so the projection MLPs can be built.

Everything is computed on the image *as it will be seen by the classifier*
(i.e. after any augmentation) and at native resolution -- never on a resized
copy, because resampling destroys exactly the traces we are looking for.

Dependencies: numpy, scipy. (No pywt / cv2 / torch required.)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Tuple

import numpy as np
from scipy.fft import dctn
from scipy.ndimage import convolve, convolve1d, uniform_filter

EPS = 1e-8
# Bump whenever a feature definition changes: cached standardisation statistics
# (train_aug_classical_single.py, classical.stats) are keyed on it and get
# recomputed instead of silently standardising new features with old moments.
FEATURE_VERSION = 2

# --------------------------------------------------------------------------- #
#  small helpers
# --------------------------------------------------------------------------- #


def _as_rgb_float(img: np.ndarray) -> np.ndarray:
    """uint8/float HxW or HxWxC  ->  float32 HxWx3 in [0, 255]."""
    a = np.asarray(img)
    if a.ndim == 2:
        a = a[:, :, None]
    if a.shape[2] == 1:
        a = np.repeat(a, 3, axis=2)
    if a.shape[2] == 4:
        a = a[:, :, :3]
    a = a.astype(np.float32)
    if a.max() <= 1.5:  # someone passed a [0,1] float image
        a = a * 255.0
    return np.ascontiguousarray(a[:, :, :3])


def _luma(rgb: np.ndarray) -> np.ndarray:
    return (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.float32)


def _sub(x: np.ndarray, max_n: int) -> np.ndarray:
    """Strided subsample -- all statistics below are population statistics, so a
    few thousand samples give a perfectly adequate estimate at a fraction of the
    cost. Stride sampling (not random) keeps this deterministic."""
    x = x.ravel()
    return x[:: max(1, x.size // max_n)] if x.size > max_n else x


def _moments(x: np.ndarray, max_n: int = 32768) -> np.ndarray:
    """[mean, std, skew, kurt] -- robust to constant input. float64: the fourth
    power of MAD-normalised coefficients overflows float32 on sparse inputs."""
    x = _sub(x, max_n).astype(np.float64, copy=False)
    m = float(x.mean(dtype=np.float64))
    d = x - m
    d2 = d * d
    v = float(d2.mean(dtype=np.float64))
    s = math.sqrt(v) + EPS
    sk = float((d2 * d).mean(dtype=np.float64)) / s**3
    ku = float((d2 * d2).mean(dtype=np.float64)) / s**4 - 3.0
    return np.array([m, math.sqrt(v), sk, ku], dtype=np.float32)


def _mad(x: np.ndarray, max_n: int = 16384, floor: float = 1e-2) -> float:
    """Robust std estimate (median absolute deviation, gaussian-consistent).

    Floored at `floor`: on a sparse input (more than half the entries exactly
    zero -- wavelet subbands of JPEG'd / blurred flat regions, residuals of
    flat patches) the MAD is 0 and everything normalised by it (std / MAD,
    spectral peak z-scores) would come out at 1e8..1e9, which no downstream
    standardisation survives. 1e-2 is far below the granularity of any
    8-bit-derived coefficient, so ordinary inputs are unaffected."""
    x = _sub(x, max_n)
    return float(max(1.4826 * np.median(np.abs(x - np.median(x))), floor) + EPS)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + EPS))


# --------------------------------------------------------------------------- #
#  patch selection  (texture-stratified, JPEG-grid aligned)
# --------------------------------------------------------------------------- #


def _integral(x: np.ndarray) -> np.ndarray:
    return np.pad(x.cumsum(0).cumsum(1), ((1, 0), (1, 0)))


def _box_means(integ: np.ndarray, ys: np.ndarray, xs: np.ndarray, ps: int) -> np.ndarray:
    Y, X = np.meshgrid(ys, xs, indexing="ij")
    s = (
        integ[Y + ps, X + ps]
        - integ[Y, X + ps]
        - integ[Y + ps, X]
        + integ[Y, X]
    )
    return s / float(ps * ps)


def select_patches(
    rgb: np.ndarray,
    patch: int = 256,
    n_rich: int = 4,
    n_poor: int = 4,
    align: int = 16,
    max_overlap: float = 0.35,
) -> List[Tuple[int, int, int, int]]:
    """
    Return a list of (y, x, patch_size, is_rich) crops.

    Two stratified pools are used, following the observation that generative
    fingerprints are *not* uniformly distributed over an image:
      * poor-texture regions (sky, skin, bokeh) expose the generator's noise
        floor / lack of sensor noise;
      * rich-texture regions expose the difference in inter-pixel correlation
        that generators cannot reproduce.
    Crop origins are snapped to a multiple of 16 so that the JPEG 8x8 (and 4:2:0
    16x16) grid inside the crop stays aligned with the crop border -- otherwise
    every DCT / blockiness feature below is destroyed.
    """
    H, W, _ = rgb.shape
    ps = min(patch, (min(H, W) // align) * align)
    ps = max(ps, 64)
    if H < ps or W < ps:  # tiny image: reflect-pad once
        rgb = np.pad(rgb, ((0, max(0, ps - H)), (0, max(0, ps - W)), (0, 0)), mode="reflect")
        H, W, _ = rgb.shape

    # scoring is done on a 2x-decimated luma map: 4x cheaper and the ranking is
    # unchanged. Crop origins are still emitted at full resolution, aligned.
    g = _luma(rgb)[::2, ::2]
    tex = np.abs(np.diff(g, axis=0, prepend=g[:1])) + np.abs(np.diff(g, axis=1, prepend=g[:, :1]))
    clipped = ((g <= 2.0) | (g >= 253.0)).astype(np.float32)

    it, ic = _integral(tex), _integral(clipped)
    stride = max(2 * align, (ps // 2 // (2 * align)) * (2 * align))
    ps2 = ps // 2
    ys = np.arange(0, (H - ps) // 2 + 1, stride // 2)
    xs = np.arange(0, (W - ps) // 2 + 1, stride // 2)
    if len(ys) == 0:
        ys = np.array([0])
    if len(xs) == 0:
        xs = np.array([0])

    score = _box_means(it, ys, xs, ps2)
    clipfrac = _box_means(ic, ys, xs, ps2)
    valid = clipfrac < 0.4
    if not valid.any():
        valid = np.ones_like(valid)

    cand = [(float(score[i, j]), 2 * int(ys[i]), 2 * int(xs[j]))
            for i in range(len(ys)) for j in range(len(xs)) if valid[i, j]]
    cand.sort(key=lambda t: t[0])

    def _pick(pool, k, is_rich):
        out = []
        for _, y, x in pool:
            ok = True
            for (y2, x2, _, _) in out:
                iw = max(0, ps - abs(x - x2))
                ih = max(0, ps - abs(y - y2))
                if iw * ih > max_overlap * ps * ps:
                    ok = False
                    break
            if ok:
                out.append((y, x, ps, is_rich))
            if len(out) == k:
                break
        return out

    poor = _pick(cand, n_poor, 0)
    rich = _pick(cand[::-1], n_rich, 1)
    sel = poor + rich

    # pad by cycling if the image was too small to yield enough distinct crops
    i = 0
    while len(sel) < n_rich + n_poor and len(sel) > 0:
        sel.append(sel[i % len(sel)])
        i += 1
    if not sel:
        sel = [(0, 0, ps, 1)] * (n_rich + n_poor)
    return sel[: n_rich + n_poor]


# --------------------------------------------------------------------------- #
#  1. Fourier / spectral family
# --------------------------------------------------------------------------- #

N_RAD, N_ANG = 64, 36

# Normalised frequencies (cycles/pixel) where up-sampling / resampling leaves
# periodic replicas. Nyquist = 0.5.  x2 transposed-conv -> 0.5 ; x4 -> 0.25 ;
# x8 -> 0.125 ; 1.5x / 3x resamples -> 1/3, 1/6.
_GRID_FREQS: List[Tuple[float, float]] = [
    (0.5, 0.0), (0.0, 0.5), (0.5, 0.5),
    (0.25, 0.0), (0.0, 0.25), (0.25, 0.25), (0.25, 0.5), (0.5, 0.25),
    (0.125, 0.0), (0.0, 0.125), (0.125, 0.125),
    (1 / 3, 0.0), (0.0, 1 / 3), (1 / 3, 1 / 3),
    (1 / 6, 0.0), (0.0, 1 / 6),
]


# Frequency offsets at which a periodically-modulated (up-sampled / resampled)
# image copies its own spectrum.
_REPLICA_SHIFTS: List[Tuple[float, float]] = [
    (0.5, 0.0), (0.0, 0.5), (0.5, 0.5),
    (0.25, 0.0), (0.0, 0.25), (0.25, 0.25),
    (1 / 3, 0.0), (0.0, 1 / 3),
]

_WELCH_N = 128  # sub-window size for periodogram averaging


@lru_cache(maxsize=8)
def _fft_grids(h: int, w: int):
    fy = np.fft.fftshift(np.fft.fftfreq(h))[:, None]
    fx = np.fft.fftshift(np.fft.fftfreq(w))[None, :]
    r = np.sqrt(fy**2 + fx**2).astype(np.float32)
    th = (np.arctan2(np.broadcast_to(fy, (h, w)), np.broadcast_to(fx, (h, w))) % np.pi).astype(np.float32)
    win = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)

    rmax = 0.5
    rid = np.clip((r / rmax * N_RAD).astype(np.int32), 0, N_RAD - 1)
    rid[r > rmax] = -1
    aid = np.clip((th / np.pi * N_ANG).astype(np.int32), 0, N_ANG - 1)
    ring = (r > 0.15) & (r <= rmax)

    gidx = []
    for (u, v) in _GRID_FREQS:
        iy = int(round(v * h)) + h // 2
        ix = int(round(u * w)) + w // 2
        gidx.append((min(max(iy, 0), h - 1), min(max(ix, 0), w - 1)))
    return win, r, rid, aid, ring, np.array(gidx, dtype=np.int32)


def _welch_periodogram(g: np.ndarray, n: int = _WELCH_N) -> np.ndarray:
    """
    Averaged (Welch) periodogram.

    A single periodogram of one patch has ~100% relative variance per bin, which
    completely buries the narrow up-sampling peaks we are looking for. Averaging
    K half-overlapping sub-windows cuts the noise floor by ~sqrt(K) at the cost
    of frequency resolution we do not need (the peaks sit at simple rationals).
    """
    h, w = g.shape
    n = min(n, h, w)
    win, *_ = _fft_grids(n, n)
    step = max(n // 2, 1)
    ys = list(range(0, h - n + 1, step)) or [0]
    xs = list(range(0, w - n + 1, step)) or [0]
    acc = np.zeros((n, n), dtype=np.float32)
    for y in ys:
        for x in xs:
            b = g[y:y + n, x:x + n]
            b = b - b.mean()
            acc += np.abs(np.fft.fft2(b * win)) ** 2
    return np.fft.fftshift(acc / (len(ys) * len(xs)))


def _radial_profile(logP: np.ndarray, rid: np.ndarray) -> np.ndarray:
    m = rid >= 0
    idx = rid[m]
    s = np.bincount(idx, weights=logP[m].astype(np.float64), minlength=N_RAD)
    c = np.bincount(idx, minlength=N_RAD)
    prof = s / np.maximum(c, 1)
    if (c == 0).any():
        good = c > 0
        prof = np.interp(np.arange(N_RAD), np.flatnonzero(good), prof[good])
    return prof.astype(np.float32)


def spectral_features(patch_gray: np.ndarray) -> np.ndarray:
    """
    Content-normalised power-spectrum descriptor.

    Natural images have an approximately 1/f^a power spectrum; the *shape of the
    deviation* from that power law -- not the raw spectrum -- carries the
    generative fingerprint. We therefore fit the log-log power law and keep
    (a) the fit parameters, (b) the residual radial profile, (c) the angular
    profile (anisotropy), (d) z-scored peaks at the canonical up-sampling
    frequencies, (e) band energies.
    """
    P = _welch_periodogram(patch_gray)
    n = P.shape[0]
    _, r, rid, aid, ring, gidx = _fft_grids(n, n)

    logP = np.log(P + EPS).astype(np.float32)
    logP -= logP[ring].mean()  # exposure / contrast invariance

    prof = _radial_profile(logP, rid)

    lo, hi = 3, N_RAD
    lr = np.log(np.arange(lo, hi, dtype=np.float64) / N_RAD * 0.5 + EPS)
    A = np.stack([lr, np.ones_like(lr)], 1)
    slope, inter = np.linalg.lstsq(A, prof[lo:hi].astype(np.float64), rcond=None)[0]
    lr_full = np.log(np.maximum(np.arange(N_RAD), 0.5) / N_RAD * 0.5)
    rad_res = (prof - (slope * lr_full + inter)).astype(np.float32)

    base2d = np.interp(r.ravel(), np.arange(N_RAD) / N_RAD * 0.5, prof).reshape(r.shape).astype(np.float32)
    dev = logP - base2d

    m = ring
    s = np.bincount(aid[m], weights=dev[m].astype(np.float64), minlength=N_ANG)
    c = np.bincount(aid[m], minlength=N_ANG)
    ang = (s / np.maximum(c, 1)).astype(np.float32)

    # peaks, z-scored against the global fluctuation level of `dev`
    noise = _mad(dev[ring]) + EPS
    pad = 3
    devp = np.pad(dev, pad, mode="reflect")
    peaks = np.empty(len(_GRID_FREQS), dtype=np.float32)
    for k, (iy, ix) in enumerate(gidx):
        win7 = devp[iy: iy + 2 * pad + 1, ix: ix + 2 * pad + 1]
        peaks[k] = (win7[pad - 1: pad + 2, pad - 1: pad + 2].max() - np.median(win7)) / noise

    # spectral replica correlation: periodic gain modulation (checkerboard from
    # transposed convs, nearest/bilinear up-sampling, resampling) copies the
    # spectrum to f + df. A delta-peak detector misses this because the replica
    # of a broadband image is broadband; correlating the whitened spectrum with
    # a shifted copy of itself detects it directly, and is what actually
    # survives JPEG.
    rep = np.empty(len(_REPLICA_SHIFTS), dtype=np.float32)
    dvr = dev * ring
    for k, (u, v) in enumerate(_REPLICA_SHIFTS):
        d2 = np.roll(np.roll(dvr, int(round(v * n)), 0), int(round(u * n)), 1)
        mm = ring & (np.roll(np.roll(ring, int(round(v * n)), 0), int(round(u * n)), 1))
        rep[k] = _corr(dev[mm], d2[mm]) if mm.sum() > 64 else 0.0

    bands = []
    edges = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    for a, b in zip(edges[:-1], edges[1:]):
        mm = (r > a) & (r <= b)
        bands.append(float(dev[mm].mean()) if mm.any() else 0.0)

    return np.concatenate(
        [
            np.array([slope, inter], dtype=np.float32),
            rad_res,
            ang,
            peaks,
            rep,
            np.array(bands, dtype=np.float32),
            np.array([float(noise), float(np.percentile(dev, 99.5) / noise)], dtype=np.float32),
        ]
    ).astype(np.float32)


D_SPEC = 2 + N_RAD + N_ANG + len(_GRID_FREQS) + len(_REPLICA_SHIFTS) + 6 + 2


# --------------------------------------------------------------------------- #
#  2. Block-DCT / JPEG family
# --------------------------------------------------------------------------- #

_ZIGZAG = np.array(
    [0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5,
     12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28,
     35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
     58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63]
)
_Q_CANDIDATES = np.arange(1, 17, dtype=np.float64)
_N_ZZ_Q = 10  # how many AC frequencies get their own quantiser estimate


def _blocks_8x8(g: np.ndarray) -> np.ndarray:
    h, w = (g.shape[0] // 8) * 8, (g.shape[1] // 8) * 8
    b = g[:h, :w].reshape(h // 8, 8, w // 8, 8).transpose(0, 2, 1, 3).reshape(-1, 8, 8)
    return b


def _quant_scores(C: np.ndarray, freqs: np.ndarray, max_n: int = 2048) -> np.ndarray:
    """
    For every AC frequency in `freqs` and every candidate step q, return
    |E[exp(2*pi*i*c/q)]|  ->  (len(freqs), len(_Q_CANDIDATES)).

    The statistic is ~1 when the coefficients of that frequency live on a
    lattice of spacing q (i.e. the image carries a JPEG quantiser with that
    step) and ~0 for continuous-valued coefficients. It is a cheap, robust
    probe of the quantisation table -- and therefore of compression history --
    that needs no access to the JPEG container.
    """
    c = C[:, freqs].astype(np.float32)
    if c.shape[0] > max_n:
        c = c[:: max(1, c.shape[0] // max_n)]
    ph = (2.0 * np.pi / _Q_CANDIDATES.astype(np.float32))[:, None, None] * c.T[None, :, :]
    sc = np.hypot(np.cos(ph).mean(2), np.sin(ph).mean(2))  # (Q, F)
    return sc.T.astype(np.float32)


def dct_features(patch_gray: np.ndarray) -> np.ndarray:
    """
    8x8 block-DCT statistics on the *aligned* JPEG grid.

    Motivation: (i) the per-frequency log-magnitude spectrum of block DCT
    coefficients separates generators from cameras far better than the pixel
    histogram (Frank et al., 2020); (ii) the sparsity pattern and the lattice
    spacing of the coefficients encode the compression history -- real photos
    are usually multiply compressed by a camera pipeline plus a platform, while
    generated images are typically compressed once, and the *mismatch* between
    the estimated quantiser and the blockiness grid reveals recompression.
    """
    b = _blocks_8x8(patch_gray - 128.0)
    C = dctn(b, axes=(1, 2), norm="ortho").reshape(len(b), 64)

    lm = np.log1p(np.abs(C))
    f_mean = lm.mean(0).astype(np.float32)                       # 64
    f_std = lm.std(0).astype(np.float32)                         # 64
    f_zero = (np.abs(C) < 0.5).mean(0).astype(np.float32)        # 64

    zz = _ZIGZAG[1: 1 + _N_ZZ_Q]
    qsc = _quant_scores(C, zz)                                   # (10, 16)
    q_best = _Q_CANDIDATES[np.argmax(qsc, 1)].astype(np.float32)  # 10
    q_str = qsc.max(1).astype(np.float32)                        # 10
    q_prof = qsc.mean(0).astype(np.float32)                      # 16

    # blockiness as a function of grid phase: strong contrast between phase 0
    # and the others == a JPEG grid aligned with the crop; a peak at another
    # phase == the image was cropped/resized after compression.
    g = patch_gray
    dh = np.abs(np.diff(g, axis=1))
    dv = np.abs(np.diff(g, axis=0))
    ph_h = np.array([dh[:, i::8].mean() for i in range(8)], dtype=np.float32)
    ph_v = np.array([dv[i::8, :].mean() for i in range(8)], dtype=np.float32)
    ph_h = ph_h / (ph_h.mean() + EPS)
    ph_v = ph_v / (ph_v.mean() + EPS)

    return np.concatenate([f_mean, f_std, f_zero, q_best, q_str, q_prof, ph_h, ph_v]).astype(np.float32)


D_DCT = 64 * 3 + _N_ZZ_Q * 2 + len(_Q_CANDIDATES) + 16


# --------------------------------------------------------------------------- #
#  3. Noise-residual family (SRM-lite co-occurrences + autocorrelation)
# --------------------------------------------------------------------------- #

_SRM_FILTERS = {
    "d1": np.array([[0, 0, 0], [0, -1, 1], [0, 0, 0]], dtype=np.float32),
    "d2": np.array([[0, 0, 0], [1, -2, 1], [0, 0, 0]], dtype=np.float32),
    "sq": np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]], dtype=np.float32) / 4.0,
    "kb": np.array(
        [[-1, 2, -2, 2, -1],
         [2, -6, 8, -6, 2],
         [-2, 8, -12, 8, -2],
         [2, -6, 8, -6, 2],
         [-1, 2, -2, 2, -1]], dtype=np.float32) / 12.0,
}
_T = 2                      # truncation -> alphabet of 2T+1 = 5 symbols
_COOC = (2 * _T + 1) ** 3   # 125 bins per filter
_LAG = 8                    # autocorrelation lag window


def _cooccurrence(q: np.ndarray) -> np.ndarray:
    s = 2 * _T + 1
    a = (q + _T).astype(np.int64)
    hh = a[:, :-2] * s * s + a[:, 1:-1] * s + a[:, 2:]
    vv = a[:-2, :] * s * s + a[1:-1, :] * s + a[2:, :]
    f = (np.bincount(hh.ravel(), minlength=s**3) + np.bincount(vv.ravel(), minlength=s**3)).astype(np.float32)
    return f / (f.sum() + EPS)


def _autocorr(x: np.ndarray, lag: int = _LAG) -> np.ndarray:
    x = x - x.mean()
    F = np.fft.rfft2(x)
    ac = np.fft.irfft2(F * np.conj(F), s=x.shape)
    ac = ac / (ac.flat[0] + EPS)
    top = np.concatenate([ac[:lag, :lag], ac[:lag, -lag:]], axis=1)  # lags 0..7 and -8..-1
    return top.astype(np.float32).ravel()  # (lag, 2*lag) -> 128


def residual_features(patch_rgb: np.ndarray) -> np.ndarray:
    """
    High-pass residual statistics.

    * Co-occurrence histograms of quantised residuals (a compact SRM / SPAM
      model) capture the local dependency structure of the noise floor: sensor
      noise + demosaicing produce a very different symbol distribution from
      decoder noise. Residuals are normalised by their own MAD before
      quantisation, which makes the descriptor invariant to global contrast and
      to the *amount* of noise, so it degrades gracefully under JPEG.
    * The 2-D autocorrelation of the residual, and of its squared envelope, is
      the single most direct detector of the periodic grid left by transposed /
      nearest-neighbour up-sampling in GAN and VAE decoders: a real photo's
      residual autocorrelation decays monotonically, a synthesised one shows
      spikes at lags 2, 4, 8.
    """
    g = _luma(patch_rgb)
    out = [np.array([], dtype=np.float32)]

    r_ref = None
    for name, k in _SRM_FILTERS.items():
        r = convolve(g, k, mode="reflect")
        sc = _mad(r)
        q = np.clip(np.rint(r / sc), -_T, _T).astype(np.int8)
        out.append(_cooccurrence(q))
        out.append(np.array([np.log1p(sc)], dtype=np.float32))
        if name == "kb":
            r_ref = r / (sc + EPS)

    # periodicity of the residual and of its energy envelope
    out.append(_autocorr(r_ref))
    env = np.abs(r_ref)
    out.append(_autocorr(env - uniform_filter(env, 9)))

    # marginal shape of the residual + per-channel noise anisotropy
    out.append(_moments(np.clip(r_ref, -8, 8)))
    ch = []
    for c in range(3):
        rc = convolve(patch_rgb[:, :, c], _SRM_FILTERS["sq"], mode="reflect")
        ch.append(np.log1p(_mad(rc)))
    out.append(np.array(ch, dtype=np.float32))
    return np.concatenate(out).astype(np.float32)


D_RES = len(_SRM_FILTERS) * (_COOC + 1) + 2 * (_LAG * 2 * _LAG) + 4 + 3


# --------------------------------------------------------------------------- #
#  4. Wavelet family (Farid-Lyu style)
# --------------------------------------------------------------------------- #

_H = np.array(
    [0.23037781330885523, 0.7148465705525415, 0.6308807679295904, -0.02798376941698385,
     -0.18703481171888114, 0.030841381835986965, 0.032883011666982945, -0.010597401784997278],
    dtype=np.float64,
)
_DEC_LO = _H[::-1].astype(np.float32)
_DEC_HI = np.array([(-1) ** k * _H[k] for k in range(len(_H))], dtype=np.float32)
_WAV_LEVELS = 3


def _dwt2(a: np.ndarray):
    h, w = (a.shape[0] // 2) * 2, (a.shape[1] // 2) * 2
    a = a[:h, :w]
    lo = convolve1d(a, _DEC_LO, axis=1, mode="wrap")[:, ::2]
    hi = convolve1d(a, _DEC_HI, axis=1, mode="wrap")[:, ::2]
    ll = convolve1d(lo, _DEC_LO, axis=0, mode="wrap")[::2]
    lh = convolve1d(lo, _DEC_HI, axis=0, mode="wrap")[::2]
    hl = convolve1d(hi, _DEC_LO, axis=0, mode="wrap")[::2]
    hh = convolve1d(hi, _DEC_HI, axis=0, mode="wrap")[::2]
    return ll, (lh, hl, hh)


def _pred_error_moments(sub: np.ndarray, parent: np.ndarray | None, sibs) -> np.ndarray:
    """
    Farid & Lyu's second half: how predictable is a coefficient magnitude from
    its spatial neighbours, its coarser-scale parent and the other orientations?
    Natural images have a very specific amount of residual unpredictability;
    generators systematically over- or under-shoot it.
    """
    m = np.log(np.abs(sub) + 1.0)
    feats = [
        np.roll(m, 1, 0), np.roll(m, -1, 0), np.roll(m, 1, 1), np.roll(m, -1, 1),
        np.log(np.abs(sibs[0]) + 1.0), np.log(np.abs(sibs[1]) + 1.0),
    ]
    if parent is not None:
        p = np.log(np.abs(parent) + 1.0)
        p = np.repeat(np.repeat(p, 2, 0), 2, 1)[: m.shape[0], : m.shape[1]]
        if p.shape == m.shape:
            feats.append(p)
    step = max(1, m.size // 8192)
    X = np.stack([f.ravel()[::step] for f in feats] + [np.ones(m.ravel()[::step].shape, np.float32)], 1)
    y = m.ravel()[::step]
    # Normal equations in float64 with a ridge RELATIVE to the diagonal. Whenever two regressors
    # coincide -- a subband constant along one axis (gradients, straight edges, i.e. the poor-texture
    # crops select_patches looks for), a checkerboard, an image < 2 px wide -- X.T @ X is exactly
    # singular, and the former absolute 1e-4 ridge vanished against float32 diagonals of 1e5..1e6
    # (np.linalg.solve: "Singular matrix"). 1e-6 of the mean diagonal keeps the system positive
    # definite in float64, perturbs w by ~1e-6 relative and leaves the residual -- all that is used
    # -- unchanged; lstsq (SVD, never singular, but 8x slower) only as a last resort.
    X64, y64 = X.astype(np.float64), y.astype(np.float64)
    XtX = X64.T @ X64
    XtX += 1e-6 * (np.trace(XtX) / XtX.shape[0] + EPS) * np.eye(XtX.shape[0])
    try:
        w = np.linalg.solve(XtX, X64.T @ y64)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(X64, y64, rcond=None)[0]
    return _moments(y64 - X64 @ w)


def wavelet_features(patch_rgb: np.ndarray) -> np.ndarray:
    g = _luma(patch_rgb)
    subs, cur = [], g
    for _ in range(_WAV_LEVELS):
        cur, det = _dwt2(cur)
        subs.append(det)

    out = []
    for lvl, (lh, hl, hh) in enumerate(subs):
        parent = subs[lvl + 1] if lvl + 1 < len(subs) else None
        for oi, s in enumerate((lh, hl, hh)):
            sc = _mad(s)
            out.append(_moments(s / sc))                       # 4  (scale-invariant shape)
            out.append(np.array([np.log1p(sc)], dtype=np.float32))
            sibs = [x for j, x in enumerate((lh, hl, hh)) if j != oi]
            out.append(_pred_error_moments(s, None if parent is None else parent[oi], sibs))  # 4
        out.append(np.array([_corr(np.abs(lh), np.abs(hl)),
                             _corr(np.abs(lh), np.abs(hh)),
                             _corr(np.abs(hl), np.abs(hh))], dtype=np.float32))

    # cross-scale energy decay: generators typically lose high-frequency energy
    e = [float(np.mean(np.abs(np.concatenate([s.ravel() for s in d])))) for d in subs]
    out.append(np.log1p(np.array(e, dtype=np.float32)))
    out.append(np.array([np.log((e[i] + EPS) / (e[i + 1] + EPS)) for i in range(len(e) - 1)], dtype=np.float32))

    # per-channel HF energy ratio (chroma is where diffusion VAEs are weakest)
    ce = []
    for c in range(3):
        _, (a, b, cc) = _dwt2(patch_rgb[:, :, c])
        ce.append(float(np.mean(np.abs(a)) + np.mean(np.abs(b)) + np.mean(np.abs(cc))))
    ce = np.array(ce, dtype=np.float32)
    out.append(np.log1p(ce))
    out.append(np.array([ce[0] / (ce[1] + EPS), ce[2] / (ce[1] + EPS)], dtype=np.float32))
    return np.concatenate(out).astype(np.float32)


D_WAV = _WAV_LEVELS * (3 * (4 + 1 + 4) + 3) + _WAV_LEVELS + (_WAV_LEVELS - 1) + 3 + 2


# --------------------------------------------------------------------------- #
#  5. Colour / CFA family
# --------------------------------------------------------------------------- #

_CFA_K = np.array([[0.25, 0.5, 0.25], [0.5, 0.0, 0.5], [0.25, 0.5, 0.25]], dtype=np.float32)


def color_features(patch_rgb: np.ndarray) -> np.ndarray:
    """
    Colour-filter-array and inter-channel evidence.

    A real photograph is demosaiced: two thirds of the pixels in R and B (half
    in G) are interpolated, so the interpolation residual has period-2 structure
    that is *phase-locked* to the sensor lattice. Generative decoders synthesise
    all three channels jointly, so their residual energy is flat across the four
    (i%2, j%2) sub-lattices. Also: camera noise is decorrelated across channels
    after demosaicing in a characteristic way, whereas decoder noise is strongly
    correlated across R/G/B.
    """
    out = []
    res = []
    for c in range(3):
        x = patch_rgb[:, :, c]
        r = x - convolve(x, _CFA_K, mode="reflect")
        res.append(r)
        v = np.array([r[i::2, j::2].var() for i in range(2) for j in range(2)], dtype=np.float32)
        v = v / (v.sum() + EPS)
        out.append(v)                                     # 4 : sub-lattice energy split
        out.append(np.array([v.max() / (v.min() + EPS)], dtype=np.float32))
    out.append(np.array([_corr(res[0], res[1]), _corr(res[1], res[2]), _corr(res[0], res[2])], dtype=np.float32))

    # opponent-colour high-frequency balance
    rg = patch_rgb[:, :, 0] - patch_rgb[:, :, 1]
    by = patch_rgb[:, :, 2] - 0.5 * (patch_rgb[:, :, 0] + patch_rgb[:, :, 1])
    y = _luma(patch_rgb)
    hf = lambda z: float(np.mean(np.abs(convolve(z, _SRM_FILTERS["sq"], mode="reflect"))))
    hy = hf(y) + EPS
    out.append(np.array([np.log1p(hy), hf(rg) / hy, hf(by) / hy], dtype=np.float32))

    # dynamic-range / clipping signature
    clip = [(patch_rgb[:, :, c] >= 253).mean() for c in range(3)] + [(patch_rgb[:, :, c] <= 2).mean() for c in range(3)]
    out.append(np.array(clip, dtype=np.float32))
    out.append(np.array([patch_rgb[:, :, c].std() for c in range(3)], dtype=np.float32) / 64.0)
    return np.concatenate(out).astype(np.float32)


D_COL = 3 * 5 + 3 + 3 + 6 + 3


# --------------------------------------------------------------------------- #
#  6. Global degradation descriptor (one token)
# --------------------------------------------------------------------------- #


def degradation_features(rgb: np.ndarray) -> np.ndarray:
    """
    An explicit, low-dimensional estimate of *how the image was degraded*:
    noise level, blur, JPEG quality, resolution, resampling.

    Two reasons to feed this to the pooler as its own token:
      1. it lets the attention pooler discount the frequency-domain tokens when
         the evidence has been destroyed (heavy JPEG / blur) and up-weight them
         when the image is clean -- i.e. learned, degradation-conditional
         gating instead of a single compromise decision rule;
      2. it isolates the main shortcut in this task into one token that can be
         ablated, dropped out, or fed through a gradient-reversal head to
         *prevent* the classifier from using "compressed ==> real".
    """
    Hf, Wf = rgb.shape[:2]
    gfull = _luma(rgb)

    # everything expensive is measured on a central, JPEG-grid-aligned window;
    # 512x512 is far more than enough for noise/blur/quantiser estimation.
    S = min(384, (min(Hf, Wf) // 16) * 16)
    S = max(S, 64)
    y0 = max(0, ((Hf - S) // 2 // 16) * 16)  # max: a side < S (32 px thumbnails) gave a negative
    x0 = max(0, ((Wf - S) // 2 // 16) * 16)  # origin, i.e. a 16 px corner instead of the whole image
    g = np.ascontiguousarray(gfull[y0:y0 + S, x0:x0 + S])
    H, W = Hf, Wf

    _, (lh, hl, hh) = _dwt2(g)
    sigma = _mad(hh)                                   # Donoho noise estimate

    lap = convolve(g, _SRM_FILTERS["sq"], mode="reflect")
    blur1 = float(lap.var())

    dh = np.abs(np.diff(g, axis=1))
    dv = np.abs(np.diff(g, axis=0))
    gm = float(dh.mean() + dv.mean())
    ph_h = np.array([dh[:, i::8].mean() for i in range(8)])
    ph_v = np.array([dv[i::8, :].mean() for i in range(8)])
    block = float(ph_h[7] / (ph_h.mean() + EPS) + ph_v[7] / (ph_v.mean() + EPS)) / 2.0

    C = dctn(_blocks_8x8(g - 128.0), axes=(1, 2), norm="ortho").reshape(-1, 64)
    zz = _ZIGZAG[1:11]
    qs = _quant_scores(C, zz)
    q_step = float(np.median(_Q_CANDIDATES[np.argmax(qs, 1)]))
    q_conf = float(qs.max(1).mean())
    zero_ac = float((np.abs(C[:, zz]) < 0.5).mean())

    hf_ratio = float((np.abs(lh).mean() + np.abs(hl).mean() + np.abs(hh).mean()) / (gm + EPS))

    return np.array(
        [
            np.log1p(sigma), np.log1p(blur1), np.log1p(gm), hf_ratio,
            block, np.log1p(q_step), q_conf, zero_ac,
            np.log(H) / 8.0, np.log(W) / 8.0, W / (H + EPS), np.log1p(H * W) / 16.0,
            float(gfull.mean()) / 255.0, float(gfull.std()) / 64.0,
            float((gfull >= 253).mean()), float((gfull <= 2).mean()),
        ],
        dtype=np.float32,
    )


D_GLOB = 16


# --------------------------------------------------------------------------- #
#  the extractor
# --------------------------------------------------------------------------- #


@dataclass
class ClassicalFeatureExtractor:
    patch: int = 256
    n_rich: int = 4
    n_poor: int = 4
    families: Tuple[str, ...] = ("spectral", "dct", "residual", "wavelet", "color")

    @property
    def n_patches(self) -> int:
        return self.n_rich + self.n_poor

    def dims(self) -> Dict[str, int]:
        d = {"spectral": D_SPEC, "dct": D_DCT, "residual": D_RES, "wavelet": D_WAV, "color": D_COL}
        out = {k: d[k] for k in self.families}
        out["global"] = D_GLOB
        return out

    def __call__(self, img: np.ndarray) -> Dict[str, np.ndarray]:
        rgb = _as_rgb_float(img)
        crops = select_patches(rgb, self.patch, self.n_rich, self.n_poor)
        H, W, _ = rgb.shape

        acc: Dict[str, List[np.ndarray]] = {k: [] for k in self.families}
        meta = []
        for (y, x, ps, is_rich) in crops:
            p = rgb[y:y + ps, x:x + ps]
            if p.shape[0] != ps or p.shape[1] != ps:
                p = np.pad(p, ((0, ps - p.shape[0]), (0, ps - p.shape[1]), (0, 0)), mode="reflect")
            pg = _luma(p)
            if "spectral" in acc:
                acc["spectral"].append(spectral_features(pg))
            if "dct" in acc:
                acc["dct"].append(dct_features(pg))
            if "residual" in acc:
                acc["residual"].append(residual_features(p))
            if "wavelet" in acc:
                acc["wavelet"].append(wavelet_features(p))
            if "color" in acc:
                acc["color"].append(color_features(p))
            meta.append([float(is_rich), y / max(H - ps, 1), x / max(W - ps, 1), np.log(ps) / 8.0])

        out = {k: np.stack(v).astype(np.float32) for k, v in acc.items()}
        out["global"] = degradation_features(rgb)
        out["patch_meta"] = np.asarray(meta, dtype=np.float32)
        for k, v in out.items():
            np.nan_to_num(v, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return out
