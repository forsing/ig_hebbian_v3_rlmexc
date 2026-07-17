from __future__ import annotations

# IG = Information Geometry (informaciona geometrija) 

"""
Geometrija prostora kombinacija — v3 RLM + ekscitacija

v1 Hebbian D. v2 Perez·circ.
v3: RLM (ne LLM) — rekurzivni lokalni koraci mase na D;
    ekscitacija drži observabilnost (mali šum / perturbacija).

Stanјe s^{(0)} = uniformna masa na last (1/7).
Korak:
  s' = s · D          (Hebbian transport energije)
  s' ← (1−ε) s' + ε · excite(s')   (ekscitacija)
  s ← s' / ||s'||₁

Posle K koraka: skor = s^{(K)} · L_perez · circ
Ban last; next. CSV: loto7_4652_k57.csv, seed=39.
Ime: ig_hebbian_v3_rlmexc.py
"""

import csv
from itertools import combinations
from math import cos, exp, pi
from pathlib import Path

import numpy as np

SEED = 39
FRONT_N = 39
FRONT_SELECT = 7
LAMBDA_TEMP = 0.35
K_RLM = 5
EPS_EXC = 0.08
ZENITH = 20.0
PEREZ_A = 4.0
PEREZ_B = 0.6
PEREZ_C = 1.2
PEREZ_D = 2.5
CIRC_PERIOD = 39
CIRC_KAPPA = 0.25
CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "loto7_4652_k57.csv"

np.random.seed(SEED)


def load_draws(csv_path: Path = CSV_PATH) -> np.ndarray:
    draws = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.reader(f):
            if len(row) < FRONT_SELECT:
                continue
            try:
                draw = sorted(int(x.strip()) for x in row[:FRONT_SELECT])
            except ValueError:
                continue
            if len(draw) == FRONT_SELECT and all(1 <= x <= FRONT_N for x in draw):
                if len(set(draw)) == FRONT_SELECT:
                    draws.append(draw)
    if not draws:
        raise ValueError(f"Nema validnih kola u {csv_path}")
    return np.array(draws, dtype=int)


def hebbian_weights(draws: np.ndarray, lam: float = LAMBDA_TEMP) -> np.ndarray:
    W = np.zeros((FRONT_N, FRONT_N), dtype=float)
    for d in draws:
        idx = [int(x) - 1 for x in d.tolist()]
        for a, b in combinations(idx, 2):
            W[a, b] += 1.0
            W[b, a] += 1.0
    for t in range(len(draws) - 1):
        a_idx = [int(x) - 1 for x in draws[t].tolist()]
        b_idx = [int(x) - 1 for x in draws[t + 1].tolist()]
        for a in a_idx:
            for b in b_idx:
                if a == b:
                    continue
                W[a, b] += lam
                W[b, a] += lam
    np.fill_diagonal(W, 0.0)
    return W


def energy_distribution(W: np.ndarray) -> np.ndarray:
    D = W.copy()
    row = D.sum(axis=1, keepdims=True)
    row = np.where(row < 1e-18, 1.0, row)
    return D / row


def perez_luminance(sun: float) -> np.ndarray:
    L = np.zeros(FRONT_N)
    for i in range(FRONT_N):
        n = i + 1
        gamma = abs(n - sun) / float(FRONT_N)
        theta = abs(n - ZENITH) / float(FRONT_N)
        L[i] = (1.0 + PEREZ_C * exp(-PEREZ_A * gamma * gamma)) * (
            1.0 + PEREZ_B * exp(-PEREZ_D * theta * theta)
        )
    return L / L.sum()


def circadian_field(t_index: int) -> np.ndarray:
    phi = 2.0 * pi * (t_index % CIRC_PERIOD) / float(CIRC_PERIOD)
    circ = np.zeros(FRONT_N)
    for i in range(FRONT_N):
        circ[i] = 1.0 + CIRC_KAPPA * cos(phi + 2.0 * pi * i / float(FRONT_N))
    return circ


def excite(s: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Mikro-ekscitacija: mali pozitivni šum proporcijonalno √s
    (budi slabe kanale — observabilnost).
    """
    noise = rng.random(FRONT_N) * np.sqrt(np.clip(s, 1e-12, None))
    e = s + noise
    return e / e.sum()


def rlm_walk(
    D: np.ndarray,
    last: np.ndarray,
    k: int = K_RLM,
    eps: float = EPS_EXC,
    seed: int = SEED,
) -> np.ndarray:
    """Rekurzivni lokalni koraci mase na Hebbian D + ekscitacija."""
    rng = np.random.default_rng(seed)
    s = np.zeros(FRONT_N)
    for x in last.tolist():
        s[int(x) - 1] = 1.0 / FRONT_SELECT
    for _ in range(k):
        s = s @ D
        s = np.clip(s, 0.0, None)
        if s.sum() <= 0:
            s = np.ones(FRONT_N) / FRONT_N
        else:
            s = s / s.sum()
        s_exc = excite(s, rng)
        s = (1.0 - eps) * s + eps * s_exc
        s = s / s.sum()
    return s


def number_scores(
    s: np.ndarray, L: np.ndarray, circ: np.ndarray, ban: set[int]
) -> dict[int, float]:
    out = {}
    for i in range(FRONT_N):
        n = i + 1
        if n in ban:
            out[n] = -1e18
        else:
            out[n] = float(s[i] * L[i] * circ[i])
    return out


def _combo_fit(combo, score, target_sum, pos_means, target_odd, ban):
    nums = sorted(combo)
    if any(x in ban for x in nums):
        return -1e18
    s = sum(score[x] for x in nums)
    s -= 0.08 * abs(sum(nums) - target_sum)
    s -= 0.04 * sum(abs(nums[i] - pos_means[i]) for i in range(FRONT_SELECT))
    odd = sum(1 for x in nums if x % 2)
    s -= 0.3 * abs(odd - target_odd)
    return s


def predict_next(draws, score, ban):
    ranked = sorted((n for n in score if n not in ban), key=lambda n: (-score[n], n))
    target_sum = float(draws.sum(axis=1).mean())
    pos_means = [float(draws[:, i].mean()) for i in range(FRONT_SELECT)]
    target_odd = float(np.mean([sum(1 for x in d if x % 2) for d in draws]))
    candidates = [sorted(ranked[:FRONT_SELECT])]
    for start in range(0, min(20, len(ranked) - FRONT_SELECT + 1)):
        candidates.append(sorted(ranked[start : start + FRONT_SELECT]))
    best, best_fit = None, -1e18
    for base in candidates:
        fit = _combo_fit(base, score, target_sum, pos_means, target_odd, ban)
        if fit > best_fit:
            best_fit, best = fit, list(base)
        for i in range(FRONT_SELECT):
            for repl in ranked[:30]:
                cand = sorted(set(base[:i] + base[i + 1 :] + [repl]))
                if len(cand) != FRONT_SELECT:
                    continue
                fit = _combo_fit(cand, score, target_sum, pos_means, target_odd, ban)
                if fit > best_fit:
                    best_fit, best = fit, cand
    return best if best is not None else sorted(ranked[:FRONT_SELECT])


def run_v3(csv_path: Path = CSV_PATH) -> None:
    draws = load_draws(csv_path)
    last = draws[-1]
    ban = set(int(x) for x in last.tolist())
    t_now = len(draws) - 1
    sun = float(np.mean(last))
    W = hebbian_weights(draws)
    D = energy_distribution(W)
    s_k = rlm_walk(D, last)
    L = perez_luminance(sun)
    circ = circadian_field(t_now)
    score = number_scores(s_k, L, circ, ban)
    combo = predict_next(draws, score, ban)

    # observabilnost: ||s_K − uniform||
    rho = float(np.linalg.norm(s_k - 1.0 / FRONT_N))

    print(f"CSV: {csv_path.name}")
    print(
        f"Kola: {len(draws)} | seed={SEED} | K={K_RLM} ε={EPS_EXC} | ig_hebbian_v3 RLM"
    )
    print(f"last: {last.tolist()}")
    print()
    print("=== RLM + ekscitacija ===")
    print(
        {
            "rho_obs": round(rho, 6),
            "s_entropy": round(float(-(s_k * np.log(s_k + 1e-18)).sum()), 4),
            "sun": round(sun, 4),
        }
    )
    print()
    ranked = sorted(
        ((n, float(score[n])) for n in range(1, FRONT_N + 1) if n not in ban),
        key=lambda t: (-t[1], t[0]),
    )
    print("=== top12 skor (s_K · L · circ) ===")
    print([(n, round(sc, 6)) for n, sc in ranked[:12]])
    print()
    print("=== next (ig_hebbian_v3) ===")
    print("next:", combo)


if __name__ == "__main__":
    run_v3()



"""
CSV: loto7_4652_k57.csv
Kola: 4652 | seed=39 | K=5 ε=0.08 | ig_hebbian_v3 RLM
last: [7, 8, 14, 15, 17, 23, 32]

=== RLM + ekscitacija ===
{'rho_obs': 0.007413, 's_entropy': 3.6625, 'sun': 16.5714}

=== top12 skor (s_K · L · circ) ===
[(25, 0.000893), (22, 0.000855), (21, 0.000852), (28, 0.000839),(24, 0.000839), (26, 0.000838), (27, 0.00081), (29, 0.000791), (19, 0.000778), (18, 0.000774), (31, 0.000762), (20, 0.000721)]

=== next (ig_hebbian_v3) ===
next: [1, 11, 16, 20, 30, 31, 33]
"""



"""
RLM koraci na manifoldu + ekscitacija.
RLM: s ← sD × K koraka + ekscitacija; skor × Perez × circ → next.
"""



"""
0. Granica
Loto i.i.d. → nema prediktivnog transporta kao u 03–05.
Ovde: algoritam uči geometriju prostora kombinacija i traži putanju energije (distribucija) → next. Ne LLM.
1. Prostor
Tačka = 7-torica (ili simplex masa na {1…39}).
Manifold = geometrija naučena iz istorije CSV (sličnost / metrika među kolima), ne nametnuti Fisher/Γ.
2. „Nebo“ (Perez intuicija)
Polje „osvetljenosti“ na prostoru brojeva/zona — analog Perez (zenit / sunce / turbidnost → parametri iz podataka).
Cirkadijalni sloj: periodična modulacija polja kroz vreme (indeks kola / faza).
3. Hebbian
Jačanje veza između ko-pojavljivanja / susednih kola na manifoldu („fire together → wire together“).
Matrica / težine = lokalna geometrija.
4. RLM (ne LLM)
Rekurzivno / lokalno učenje putanje na tom grafu/manifoldu (stanje → korak → ažuriranje).
Ekscitacija: mali perturbatori da se održi observabilnost geometrije (Stošić intuicija).
5. Energija = distribucija
Cilj: pomeraj mase/energije (distribucija na simpleksu), ne rang frekvencije.
Putanja ≈ diskretni OT korak na naučenoj metriki (Hebbian+RLM), ne sirovi Sinkhorn kao „predikcija“.
6. next
Kraj putanje / maksimum energije pod zabranom last → jedna kombinacija.
Merilo: gde putanja prati empiriju vs gde odstupa → tada nadogradnja (ne novi šum).

v1 — prostor + Hebbian težine + next
v2 — Perez-polje + cirkadijalna faza
v3 — RLM koraci na manifoldu + ekscitacija
v4 — energija/distribucija OT na naučenoj metriki → next + dijagnostika odstupanja

Seed 39, CSV loto7_4650_k56, samo simulator/.
"""
