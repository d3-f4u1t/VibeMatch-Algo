export interface PythonSnippet {
  id: string;
  title: string;
  targetModule: string;
  summary: string;
  code: string;
}

export const PYTHON_REFERENCE_MODULES: PythonSnippet[] = [
  {
    id: 'vector_fixes',
    title: '01. Pairwise Shrinkage, Per-Column Centering & Track-Inclusive Taste',
    targetModule: 'app/services/vector.py',
    summary:
      'Fixes cross-column mean centering, folds tracks into taste_text, uses pair_reliability(lam_viewer, lam_cand) across layers, routes candidate-side behavior reliability properly, and removes raw cosine % from user-facing reasons.',
    code: `def build_taste_text(user) -> str:
    artists = user.artists or []
    tracks = user.tracks or []
    if not artists and not tracks:
        return ""
    artist_names = _unique([a.name for a in artists if a.name])
    tags = [t for a in artists if isinstance(getattr(a, "tags", None), list) for t in a.tags]
    track_bits = _unique([f"{t.title} by {t.artist_name}" for t in tracks if t.title])

    parts = []
    if artist_names:
        parts.append(f"Favorite artists: {', '.join(artist_names)}")
    if track_bits:
        parts.append(f"Favorite tracks: {', '.join(track_bits)}")
    genres = _unique(tags)
    if genres:
        parts.append(f"Genres and styles: {', '.join(genres)}")
    return ". ".join(parts) + "."

def _population_mean_vector(db: Session | None, column_name: str = "music_vector") -> np.ndarray | None:
    if db is None:
        return None
    User = _first_user_model()
    col = getattr(User, column_name)
    rows = db.query(col).filter(col.isnot(None)).all()
    vecs = [v for (raw,) in rows if (v := _as_vector(raw)) is not None]
    return np.mean(np.vstack(vecs), axis=0) if vecs else None

def _directional_score(viewer, candidate, db, mu: dict[str, float], w: dict[str, float]) -> tuple[float, dict]:
    layers = {
        "taste": taste_similarity(viewer, candidate, db),
        "context": context_similarity(viewer, candidate),
        "persona": persona_similarity(viewer, candidate),
        "identity": identity_similarity(viewer, candidate),
        "behavior": behavior_similarity(viewer, candidate, db),
    }
    lambdas = {
        "taste": pair_reliability(
            reliability(n_eff_taste(viewer), LAYER_HALF_LIFE["taste"]),
            reliability(n_eff_taste(candidate), LAYER_HALF_LIFE["taste"]),
        ),
        "context": pair_reliability(
            reliability(n_eff_context(viewer), LAYER_HALF_LIFE["context"]),
            reliability(n_eff_context(candidate), LAYER_HALF_LIFE["context"]),
        ),
        "persona": pair_reliability(
            reliability(n_eff_persona(viewer), LAYER_HALF_LIFE["persona"]),
            reliability(n_eff_persona(candidate), LAYER_HALF_LIFE["persona"]),
        ),
        "identity": pair_reliability(
            reliability(n_eff_identity(viewer), LAYER_HALF_LIFE["identity"]),
            reliability(n_eff_identity(candidate), LAYER_HALF_LIFE["identity"]),
        ),
        "behavior": pair_reliability(
            reliability(n_eff_taste(viewer), LAYER_HALF_LIFE["taste"]),
            reliability(n_eff_behavior(candidate, db), LAYER_HALF_LIFE["behavior"]),
        ),
    }
    total = 0.0
    detail = {}
    for layer in LAYERS:
        lam = lambdas[layer]
        raw = float(np.clip(layers[layer], 0.0, 1.0))
        shrunk = lam * raw + (1.0 - lam) * mu[layer]
        contrib = w[layer] * shrunk
        total += contrib
        detail[layer] = {"raw": round(raw, 4), "lambda": round(lam, 4), "contribution": round(contrib, 4)}
    return float(np.clip(total, 0.0, 1.0)), detail`,
  },
  {
    id: 'behavior_fixes',
    title: '02. Single-Pass O(S+M) Behavior Index & Non-Collapsing Alignment',
    targetModule: 'app/services/behavior.py',
    summary:
      'Eliminates O(S^2) DB queries per swipe, fixes zero-collapse in attraction_alignment when passes=0, and outputs selectivity damping phi(r_u).',
    code: `def selectivity_damping(right_swipe_rate: float, total_swipes: int, gamma: float = 14.0) -> float:
    if total_swipes < 15:
        return 1.0
    if 0.15 <= right_swipe_rate <= 0.55:
        return 1.0
    if right_swipe_rate > 0.55:
        excess = right_swipe_rate - 0.55
        return max(0.15, math.exp(-gamma * excess * excess))
    return 0.7 + 0.3 * (right_swipe_rate / 0.15)

def attraction_alignment(viewer_vec: np.ndarray, liked_centroid: np.ndarray | None, passed_centroid: np.ndarray | None) -> float:
    if viewer_vec.size == 0 or (liked_centroid is None and passed_centroid is None):
        return 0.5
    if liked_centroid is not None and passed_centroid is not None:
        return float(np.clip(0.5 + 0.25 * (cos(viewer_vec, liked_centroid) - cos(viewer_vec, passed_centroid)), 0.0, 1.0))
    if liked_centroid is not None:
        return float(np.clip(0.5 + 0.5 * cos(viewer_vec, liked_centroid), 0.0, 1.0))
    return float(np.clip(0.5 - 0.5 * cos(viewer_vec, passed_centroid), 0.0, 1.0))`,
  },
  {
    id: 'clustering_module',
    title: '03. Attraction-Profile DBSCAN Clustering & Quality Gates',
    targetModule: 'app/services/clustering.py',
    summary:
      'Forms overlapping Set-D (demographics), Set-M (taste), Set-B (behavior), and Set-P (DBSCAN over attraction signatures), smoothly blending cold-start to attraction pools.',
    code: `def build_attraction_signature(liked_centroid: np.ndarray | None, passed_centroid: np.ndarray | None, spread: float, right_rate: float, beta: float = 0.35):
    if liked_centroid is None:
        return None
    direction = liked_centroid - beta * passed_centroid if passed_centroid is not None else liked_centroid
    norm = np.linalg.norm(direction)
    if norm < 1e-8:
        return None
    return np.concatenate([direction / norm, [spread, right_rate]])

def passes_quality_gate(user_count: int, avg_richness: float, match_rate: float, days_inactive: int) -> bool:
    return user_count >= 10 and avg_richness >= 0.30 and match_rate >= 0.10 and days_inactive <= 90`,
  },
  {
    id: 'gale_shapley_module',
    title: '04. Batched Many-to-Many Top-K Gale-Shapley & Exposure Budgets',
    targetModule: 'app/services/stable_match.py',
    summary:
      'Runs exact Many-to-Many Deferred Acceptance on bipartite pools and cloned capacity-K matching with blocking-pair repair on mixed orientation pools, respecting exposure budgets and per-set Elo.',
    code: `def expected_acceptance(reverse_compat: float, elo_proposer: float, elo_receiver: float, selectivity_r: float, elo_sens: float = 0.75) -> float:
    elo_gap = max(0.0, elo_receiver - elo_proposer) / 400.0
    logit = 3.2 * (reverse_compat - 0.45) - elo_sens * elo_gap + 0.6 * (selectivity_r - 0.35)
    return 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, logit))))

def bipartite_many_to_many_da(proposers, pref_lists, receiver_caps, utility_lookup, q_out: int = 8):
    held = defaultdict(set)
    next_idx = {p: 0 for p in proposers}
    active = True
    while active:
        active = False
        for p in proposers:
            prefs = pref_lists[p]
            while len(held[p]) < q_out and next_idx[p] < len(prefs):
                active = True
                r = prefs[next_idx[p]]
                next_idx[p] += 1
                if r in held[p] or (r, p) not in utility_lookup:
                    continue
                if len(held[r]) < receiver_caps[r]:
                    held[r].add(p)
                    held[p].add(r)
                else:
                    worst = min(held[r], key=lambda x: (utility_lookup[(r, x)], x))
                    if utility_lookup[(r, p)] > utility_lookup[(r, worst)]:
                        held[r].remove(worst)
                        held[worst].discard(r)
                        held[r].add(p)
                        held[p].add(r)
    return held`,
  },
];
