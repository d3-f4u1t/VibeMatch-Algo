import { cosineSimilarity, EMBED_DIM, normalizeVector } from './embeddings';
import {
  CompatibilityResult,
  ConversationRecord,
  GameTheoryConfig,
  LayerDetail,
  LayerName,
  MatchDriver,
  SwipeAction,
  SwipeEvent,
  UserBehaviorStats,
  UserProfile,
} from './types';

export const LAYERS: LayerName[] = ['taste', 'context', 'persona', 'identity', 'behavior'];

export const LAYER_HALF_LIFE: Record<LayerName, number> = {
  taste: 6.0,
  context: 4.0,
  persona: 1.0,
  identity: 0.5,
  behavior: 25.0,
};

export const SIGNAL_WEIGHTS = {
  conversation_both: 3.0,
  conversation_one_side: 2.0,
  mutual_never_opened: -1.5,
  one_sided_like: -1.0,
  super_like: 1.0,
  like: 0.6,
  pass: 0.3,
  unresolved_like: 0.1,
} as const;

const INTENT_DISTANCE: Record<string, number> = {
  long_term: 0,
  friendship: 1,
  short_term: 2,
};

const DRIVER_LABELS: Record<LayerName, string> = {
  taste: 'music taste',
  context: 'listening mood & context',
  persona: 'personality & bio',
  identity: 'stage of life & city',
  behavior: 'attraction patterns',
};

export function reliability(nEff: number, halfLife: number): number {
  if (halfLife <= 0) return nEff > 0 ? 1 : 0;
  if (nEff <= 0) return 0;
  return nEff / (nEff + halfLife);
}

export function pairReliability(lamA: number, lamB: number): number {
  if (lamA <= 0 || lamB <= 0) return 0;
  return Math.sqrt(lamA * lamB);
}

export function harmonicMean(a: number, b: number): number {
  if (a <= 0 || b <= 0) return 0;
  return (2 * a * b) / (a + b);
}

export function nEffTaste(user: UserProfile): number {
  return user.artists.length + user.tracks.length;
}

export function nEffContext(user: UserProfile): number {
  return (
    user.musicMoods.length +
    user.musicEras.length +
    user.musicContexts.length +
    (user.musicEnergy.trim() ? 1 : 0)
  );
}

export function nEffPersona(user: UserProfile): number {
  const bio = user.bio.trim();
  if (!bio) return 0;
  const sentences = bio.split(/[.!?]+/).filter((s) => s.trim().length > 0).length;
  return Math.max(1, sentences / 3);
}

export function nEffIdentity(user: UserProfile): number {
  let count = 0;
  if (user.gender) count += 1;
  if (user.interestedIn.length > 0) count += 1;
  if (user.age > 0) count += 1;
  if (user.city.trim()) count += 1;
  if (user.intent) count += 1;
  return count;
}

export function selectivityDamping(
  rightSwipeRate: number,
  totalSwipes: number,
  gamma: number
): number {
  if (totalSwipes <= 6) return 1.0;
  const ramp = Math.min(1, (totalSwipes - 6) / 10);
  let target = 1.0;
  if (rightSwipeRate > 0.55) {
    const excess = rightSwipeRate - 0.55;
    target = Math.max(0.15, Math.exp(-gamma * excess * excess));
  } else if (rightSwipeRate < 0.15) {
    target = 0.7 + 0.3 * (rightSwipeRate / 0.15);
  }
  return 1.0 - ramp * (1.0 - target);
}

// Single-pass O(S + M) behavioral index (eliminates O(S^2) DB scans in repo)
export function computeAllBehaviorStats(
  users: UserProfile[],
  swipes: SwipeEvent[],
  conversations: ConversationRecord[],
  config: GameTheoryConfig
): Record<string, UserBehaviorStats> {
  const userMap = new Map<string, UserProfile>();
  for (const u of users) userMap.set(u.id, u);

  const directed = new Map<string, SwipeEvent>();
  const swipesByUser = new Map<string, SwipeEvent[]>();
  for (const s of swipes) {
    directed.set(`${s.swiperId}::${s.targetId}`, s);
    const list = swipesByUser.get(s.swiperId);
    if (list) list.push(s);
    else swipesByUser.set(s.swiperId, [s]);
  }

  const convoSpeakers = new Map<string, Set<string>>();
  for (const c of conversations) {
    const key = c.userA < c.userB ? `${c.userA}::${c.userB}` : `${c.userB}::${c.userA}`;
    convoSpeakers.set(key, new Set(c.speakers));
  }

  const stats: Record<string, UserBehaviorStats> = {};

  for (const u of users) {
    const uSwipes = swipesByUser.get(u.id) || [];
    let nEff = 0;
    let likeCount = 0;
    let passCount = 0;
    let superLikeCount = 0;

    const likedWeightedVecs: { vec: number[]; w: number }[] = [];
    const passedVecs: number[][] = [];

    for (const s of uSwipes) {
      if (s.action === 'LIKE') likeCount += 1;
      else if (s.action === 'SUPER_LIKE') superLikeCount += 1;
      else passCount += 1;

      const rev = directed.get(`${s.targetId}::${u.id}`);
      const isPositive = s.action === 'LIKE' || s.action === 'SUPER_LIKE';
      const revPositive = rev ? rev.action === 'LIKE' || rev.action === 'SUPER_LIKE' : false;
      const isMutual = isPositive && revPositive;
      const settled = s.ageDays >= config.settlingDays;

      const pairKey = u.id < s.targetId ? `${u.id}::${s.targetId}` : `${s.targetId}::${u.id}`;
      const speakers = convoSpeakers.get(pairKey);

      let weight: number;
      if (speakers && speakers.size > 0) {
        weight =
          speakers.size >= 2
            ? SIGNAL_WEIGHTS.conversation_both
            : SIGNAL_WEIGHTS.conversation_one_side;
      } else if (isMutual) {
        weight = settled
          ? SIGNAL_WEIGHTS.mutual_never_opened
          : SIGNAL_WEIGHTS.unresolved_like;
      } else if (s.action === 'SUPER_LIKE') {
        weight = SIGNAL_WEIGHTS.super_like;
      } else if (s.action === 'LIKE') {
        weight = settled
          ? SIGNAL_WEIGHTS.one_sided_like
          : SIGNAL_WEIGHTS.unresolved_like;
      } else {
        weight = SIGNAL_WEIGHTS.pass;
      }

      nEff += Math.abs(weight);

      const target = userMap.get(s.targetId);
      if (target && target.tasteVec.length === EMBED_DIM) {
        if (isPositive) {
          const outcomeBoost = speakers && speakers.size >= 2 ? 3.0 : s.action === 'SUPER_LIKE' ? 1.5 : 1.0;
          likedWeightedVecs.push({ vec: target.tasteVec, w: outcomeBoost });
        } else {
          passedVecs.push(target.tasteVec);
        }
      }
    }

    let likedCentroid: number[] | null = null;
    if (likedWeightedVecs.length > 0) {
      const acc = new Array<number>(EMBED_DIM).fill(0);
      let wSum = 0;
      for (const item of likedWeightedVecs) {
        for (let i = 0; i < EMBED_DIM; i++) acc[i] += item.vec[i] * item.w;
        wSum += item.w;
      }
      if (wSum > 0) {
        likedCentroid = normalizeVector(acc.map((v) => v / wSum));
      }
    }

    let passedCentroid: number[] | null = null;
    if (passedVecs.length > 0) {
      const acc = new Array<number>(EMBED_DIM).fill(0);
      for (const vec of passedVecs) {
        for (let i = 0; i < EMBED_DIM; i++) acc[i] += vec[i];
      }
      passedCentroid = normalizeVector(acc.map((v) => v / passedVecs.length));
    }

    let likedSpread = 0;
    if (likedCentroid && likedWeightedVecs.length > 1) {
      let distSum = 0;
      for (const item of likedWeightedVecs) {
        distSum += 1 - cosineSimilarity(item.vec, likedCentroid);
      }
      likedSpread = distSum / likedWeightedVecs.length;
    }

    const totalSwipes = uSwipes.length;
    const rightSwipeRate = totalSwipes > 0 ? (likeCount + superLikeCount) / totalSwipes : 0.35;
    const damp = selectivityDamping(rightSwipeRate, totalSwipes, config.spamDampingGamma);
    const lamB = reliability(nEff, LAYER_HALF_LIFE.behavior);

    // Attraction signature combines contrasted liked/passed direction + spread + selectivity
    let attractionSignature: number[] | null = null;
    if (likedCentroid && lamB >= 0.15) {
      const beta = 0.35;
      const rawSig = likedCentroid.map((v, i) =>
        passedCentroid ? v - beta * passedCentroid[i] : v
      );
      const normDir = normalizeVector(rawSig);
      if (normDir.length === EMBED_DIM) {
        attractionSignature = [...normDir, likedSpread, rightSwipeRate];
      }
    }

    stats[u.id] = {
      userId: u.id,
      nEff,
      lambdaBehavior: lamB,
      totalSwipes,
      likeCount,
      passCount,
      superLikeCount,
      rightSwipeRate,
      selectivityDamping: damp,
      likedCentroid,
      passedCentroid,
      likedSpread,
      attractionSignature,
    };
  }

  return stats;
}

// Fixed attraction alignment: neutral 0.5 when no passes or equal similarity (never collapses to 0)
export function attractionAlignment(
  viewer: UserProfile,
  candidateStats: UserBehaviorStats | undefined
): number {
  if (!candidateStats || viewer.tasteVec.length === 0) return 0.5;
  const { likedCentroid, passedCentroid } = candidateStats;
  if (!likedCentroid && !passedCentroid) return 0.5;

  if (likedCentroid && passedCentroid) {
    const cosLike = cosineSimilarity(viewer.tasteVec, likedCentroid);
    const cosPass = cosineSimilarity(viewer.tasteVec, passedCentroid);
    return Math.max(0, Math.min(1, 0.5 + 0.25 * (cosLike - cosPass)));
  }
  if (likedCentroid) {
    const cosLike = cosineSimilarity(viewer.tasteVec, likedCentroid);
    return Math.max(0, Math.min(1, 0.5 + 0.5 * cosLike));
  }
  const cosPass = cosineSimilarity(viewer.tasteVec, passedCentroid!);
  return Math.max(0, Math.min(1, 0.5 - 0.5 * cosPass));
}

export function sharedArtists(a: UserProfile, b: UserProfile): string[] {
  const bSet = new Set(b.artists.map((x) => x.name.trim().toLowerCase()));
  return a.artists
    .map((x) => x.name.trim())
    .filter((name) => name && bSet.has(name.toLowerCase()));
}

export function sharedTracks(a: UserProfile, b: UserProfile): string[] {
  const bSet = new Set(b.tracks.map((x) => x.title.trim().toLowerCase()));
  return a.tracks
    .map((x) => x.title.trim())
    .filter((title) => title && bSet.has(title.toLowerCase()));
}

export function tasteSimilarity(viewer: UserProfile, candidate: UserProfile): number {
  const emb = Math.max(0, cosineSimilarity(viewer.tasteVec, candidate.tasteVec));
  const aArtists = new Set(viewer.artists.map((x) => x.name.trim().toLowerCase()));
  const bArtists = new Set(candidate.artists.map((x) => x.name.trim().toLowerCase()));
  if (aArtists.size === 0 || bArtists.size === 0) return emb;

  let overlapA = 0;
  for (const name of aArtists) if (bArtists.has(name)) overlapA += 1;
  const artistContainment = overlapA / aArtists.size;

  const aTracks = new Set(viewer.tracks.map((x) => x.title.trim().toLowerCase()));
  const bTracks = new Set(candidate.tracks.map((x) => x.title.trim().toLowerCase()));
  let overlapT = 0;
  for (const t of aTracks) if (bTracks.has(t)) overlapT += 1;
  const trackContainment = aTracks.size > 0 ? overlapT / aTracks.size : 0;

  const explicit = 0.7 * artistContainment + 0.3 * trackContainment;
  return Math.max(0, Math.min(1, 0.5 * emb + 0.5 * explicit));
}

export function contextSimilarity(viewer: UserProfile, candidate: UserProfile): number {
  return Math.max(0, cosineSimilarity(viewer.contextVec, candidate.contextVec));
}

export function personaSimilarity(viewer: UserProfile, candidate: UserProfile): number {
  return Math.max(0, cosineSimilarity(viewer.personaVec, candidate.personaVec));
}

export function identitySimilarity(viewer: UserProfile, candidate: UserProfile): number {
  const ageGap = Math.abs(viewer.age - candidate.age);
  const ageScore = Math.max(0, 1 - Math.max(0, ageGap - 2) / 13);

  const cityScore =
    viewer.city.trim().toLowerCase() === candidate.city.trim().toLowerCase() ? 1.0 : 0.4;

  let intentScore = 0.75;
  if (viewer.intent !== 'open' && candidate.intent !== 'open') {
    const gap = Math.abs((INTENT_DISTANCE[viewer.intent] ?? 1) - (INTENT_DISTANCE[candidate.intent] ?? 1));
    intentScore = Math.max(0, 1 - 0.35 * gap);
  }

  return (ageScore + cityScore + intentScore) / 3;
}

export function passesHardFilters(viewer: UserProfile, candidate: UserProfile): boolean {
  if (viewer.id === candidate.id) return false;
  if (viewer.artists.length < 3 || viewer.tracks.length < 4) return false;
  if (candidate.artists.length < 3 || candidate.tracks.length < 4) return false;

  // Bidirectional orientation gate (fixes straight-only logic in repo)
  if (!viewer.interestedIn.includes(candidate.gender)) return false;
  if (!candidate.interestedIn.includes(viewer.gender)) return false;

  // Age range
  if (candidate.age < viewer.ageMin || candidate.age > viewer.ageMax) return false;
  if (viewer.age < candidate.ageMin || viewer.age > candidate.ageMax) return false;

  // Intent clash (strict unless either is open)
  if (viewer.intent !== 'open' && candidate.intent !== 'open' && viewer.intent !== candidate.intent) {
    return false;
  }

  // Dealbreakers
  const riskyValues = new Set(['regularly', 'socially', 'sometimes', 'yes', 'often']);
  for (const d of viewer.dealbreakers) {
    const val = (candidate.habits[d.toLowerCase()] || '').toLowerCase();
    if (riskyValues.has(val)) return false;
  }
  for (const d of candidate.dealbreakers) {
    const val = (viewer.habits[d.toLowerCase()] || '').toLowerCase();
    if (riskyValues.has(val)) return false;
  }

  // Tight local city gate
  if (
    viewer.maxDistanceKm <= 50 &&
    viewer.city.trim().toLowerCase() !== candidate.city.trim().toLowerCase()
  ) {
    return false;
  }

  return true;
}

export function computePopulationMeans(
  users: UserProfile[],
  behaviorStats: Record<string, UserBehaviorStats>
): Record<LayerName, number> {
  const accum: Record<LayerName, number[]> = {
    taste: [],
    context: [],
    persona: [],
    identity: [],
    behavior: [],
  };

  const sampleLimit = Math.min(users.length, 45);
  for (let i = 0; i < sampleLimit; i++) {
    for (let j = i + 1; j < Math.min(users.length, i + 12); j++) {
      const a = users[i];
      const b = users[j];
      accum.taste.push(tasteSimilarity(a, b));
      accum.context.push(contextSimilarity(a, b));
      accum.persona.push(personaSimilarity(a, b));
      accum.identity.push(identitySimilarity(a, b));
      accum.behavior.push(attractionAlignment(a, behaviorStats[b.id]));
    }
  }

  return {
    taste: accum.taste.length ? accum.taste.reduce((s, v) => s + v, 0) / accum.taste.length : 0.5,
    context: accum.context.length ? accum.context.reduce((s, v) => s + v, 0) / accum.context.length : 0.5,
    persona: accum.persona.length ? accum.persona.reduce((s, v) => s + v, 0) / accum.persona.length : 0.5,
    identity: accum.identity.length ? accum.identity.reduce((s, v) => s + v, 0) / accum.identity.length : 0.5,
    behavior: accum.behavior.length ? accum.behavior.reduce((s, v) => s + v, 0) / accum.behavior.length : 0.5,
  };
}

function mannWhitneyAuc(positives: number[], negatives: number[]): number | null {
  if (positives.length === 0 || negatives.length === 0) return null;
  let wins = 0;
  for (const p of positives) {
    for (const n of negatives) {
      if (p > n) wins += 1;
      else if (p === n) wins += 0.5;
    }
  }
  return wins / (positives.length * negatives.length);
}

// Settling-aware, deduplicated, reliability-gated empirical AUC weight learner
export function computeLayerWeights(
  users: UserProfile[],
  swipes: SwipeEvent[],
  behaviorStats: Record<string, UserBehaviorStats>,
  config: GameTheoryConfig
): Record<LayerName, number> {
  const uniform: Record<LayerName, number> = {
    taste: 0.2,
    context: 0.2,
    persona: 0.2,
    identity: 0.2,
    behavior: 0.2,
  };

  const userMap = new Map(users.map((u) => [u.id, u]));
  const directed = new Map<string, SwipeEvent>();
  for (const s of swipes) directed.set(`${s.swiperId}::${s.targetId}`, s);

  const positives: [UserProfile, UserProfile][] = [];
  const negatives: [UserProfile, UserProfile][] = [];
  const seenMutual = new Set<string>();

  for (const s of swipes) {
    if (s.action !== 'LIKE' && s.action !== 'SUPER_LIKE') continue;
    const a = userMap.get(s.swiperId);
    const b = userMap.get(s.targetId);
    if (!a || !b) continue;

    const rev = directed.get(`${s.targetId}::${s.swiperId}`);
    const revPositive = rev ? rev.action === 'LIKE' || rev.action === 'SUPER_LIKE' : false;

    if (revPositive) {
      const key = a.id < b.id ? `${a.id}::${b.id}` : `${b.id}::${a.id}`;
      if (!seenMutual.has(key)) {
        seenMutual.add(key);
        positives.push([a, b]);
      }
    } else if (s.ageDays >= config.settlingDays || (rev && rev.action === 'PASS')) {
      negatives.push([a, b]);
    }
  }

  if (positives.length < 5 || negatives.length < 10) return uniform;

  const disc: Record<LayerName, number> = {
    taste: 0,
    context: 0,
    persona: 0,
    identity: 0,
    behavior: 0,
  };
  const attributableCount: Record<LayerName, number> = {
    taste: 0,
    context: 0,
    persona: 0,
    identity: 0,
    behavior: 0,
  };

  for (const layer of LAYERS) {
    const posScores: number[] = [];
    const negScores: number[] = [];

    const evalLayer = (a: UserProfile, b: UserProfile): { score: number; lam: number } => {
      if (layer === 'taste') {
        const lam = pairReliability(
          reliability(nEffTaste(a), LAYER_HALF_LIFE.taste),
          reliability(nEffTaste(b), LAYER_HALF_LIFE.taste)
        );
        return { score: harmonicMean(tasteSimilarity(a, b), tasteSimilarity(b, a)), lam };
      }
      if (layer === 'context') {
        const lam = pairReliability(
          reliability(nEffContext(a), LAYER_HALF_LIFE.context),
          reliability(nEffContext(b), LAYER_HALF_LIFE.context)
        );
        return { score: contextSimilarity(a, b), lam };
      }
      if (layer === 'persona') {
        const lam = pairReliability(
          reliability(nEffPersona(a), LAYER_HALF_LIFE.persona),
          reliability(nEffPersona(b), LAYER_HALF_LIFE.persona)
        );
        return { score: personaSimilarity(a, b), lam };
      }
      if (layer === 'identity') {
        const lam = pairReliability(
          reliability(nEffIdentity(a), LAYER_HALF_LIFE.identity),
          reliability(nEffIdentity(b), LAYER_HALF_LIFE.identity)
        );
        return { score: identitySimilarity(a, b), lam };
      }
      const lam = pairReliability(
        reliability(nEffTaste(a), LAYER_HALF_LIFE.taste),
        behaviorStats[b.id]?.lambdaBehavior ?? 0
      );
      return { score: attractionAlignment(a, behaviorStats[b.id]), lam };
    };

    for (const [a, b] of positives) {
      const r = evalLayer(a, b);
      if (r.lam >= 0.45) posScores.push(r.score);
    }
    for (const [a, b] of negatives) {
      const r = evalLayer(a, b);
      if (r.lam >= 0.45) negScores.push(r.score);
    }

    attributableCount[layer] = posScores.length + negScores.length;
    if (posScores.length >= 4 && negScores.length >= 6) {
      const auc = mannWhitneyAuc(posScores, negScores);
      disc[layer] = auc !== null ? Math.max(0, 2 * auc - 1) : 0;
    }
  }

  const totalDisc = Object.values(disc).reduce((s, v) => s + v, 0);
  if (totalDisc <= 0) return uniform;

  const blended: Record<LayerName, number> = { ...uniform };
  for (const layer of LAYERS) {
    const emp = disc[layer] / totalDisc;
    const m = attributableCount[layer];
    if (m < 10) {
      blended[layer] = uniform[layer];
    } else {
      blended[layer] = (m * emp + config.priorStrengthKappa * uniform[layer]) / (m + config.priorStrengthKappa);
    }
  }

  const norm = Object.values(blended).reduce((s, v) => s + v, 0) || 1;
  for (const layer of LAYERS) blended[layer] /= norm;
  return blended;
}

function directionalScore(
  viewer: UserProfile,
  candidate: UserProfile,
  behaviorStats: Record<string, UserBehaviorStats>,
  weights: Record<LayerName, number>,
  mu: Record<LayerName, number>
): { score: number; details: Record<LayerName, LayerDetail> } {
  const rawScores: Record<LayerName, number> = {
    taste: tasteSimilarity(viewer, candidate),
    context: contextSimilarity(viewer, candidate),
    persona: personaSimilarity(viewer, candidate),
    identity: identitySimilarity(viewer, candidate),
    behavior: attractionAlignment(viewer, behaviorStats[candidate.id]),
  };

  // Correct pairwise reliability: symmetric layers use geometric pairReliability;
  // candidate-attraction behavior layer requires candidate's behavior evidence + viewer's taste.
  const lambdas: Record<LayerName, number> = {
    taste: pairReliability(
      reliability(nEffTaste(viewer), LAYER_HALF_LIFE.taste),
      reliability(nEffTaste(candidate), LAYER_HALF_LIFE.taste)
    ),
    context: pairReliability(
      reliability(nEffContext(viewer), LAYER_HALF_LIFE.context),
      reliability(nEffContext(candidate), LAYER_HALF_LIFE.context)
    ),
    persona: pairReliability(
      reliability(nEffPersona(viewer), LAYER_HALF_LIFE.persona),
      reliability(nEffPersona(candidate), LAYER_HALF_LIFE.persona)
    ),
    identity: pairReliability(
      reliability(nEffIdentity(viewer), LAYER_HALF_LIFE.identity),
      reliability(nEffIdentity(candidate), LAYER_HALF_LIFE.identity)
    ),
    behavior: pairReliability(
      reliability(nEffTaste(viewer), LAYER_HALF_LIFE.taste),
      behaviorStats[candidate.id]?.lambdaBehavior ?? 0
    ),
  };

  let total = 0;
  const details = {} as Record<LayerName, LayerDetail>;

  for (const layer of LAYERS) {
    const raw = Math.max(0, Math.min(1, rawScores[layer]));
    const lam = lambdas[layer];
    const shrunk = lam * raw + (1 - lam) * mu[layer];
    const contribution = weights[layer] * shrunk;
    total += contribution;
    details[layer] = {
      raw: Number(raw.toFixed(4)),
      lambda: Number(lam.toFixed(4)),
      shrunk: Number(shrunk.toFixed(4)),
      mu: Number(mu[layer].toFixed(4)),
      weight: Number(weights[layer].toFixed(4)),
      contribution: Number(contribution.toFixed(4)),
    };
  }

  return { score: Math.max(0, Math.min(1, total)), details };
}

export function buildMatchDrivers(
  details: Record<LayerName, LayerDetail>,
  weights: Record<LayerName, number>
): MatchDriver[] {
  const drivers: MatchDriver[] = [];
  for (const layer of LAYERS) {
    const info = details[layer];
    if (!info || weights[layer] < 0.05 || info.lambda <= 0) continue;
    drivers.push({
      factor: layer,
      label: DRIVER_LABELS[layer],
      weight: Number(weights[layer].toFixed(4)),
      rawScore: info.raw,
      reliability: info.lambda,
      impact: Number((weights[layer] * info.lambda * info.raw).toFixed(4)),
    });
  }
  drivers.sort((a, b) => b.impact - a.impact);
  return drivers;
}

// Clean explainability: no uncalibrated cosine percentages leaked to users
export function buildMatchReason(
  sArtists: string[],
  sTracks: string[],
  drivers: MatchDriver[]
): string {
  const trusted = drivers.filter((d) => d.reliability >= 0.5);
  const parts: string[] = [];
  if (sTracks.length > 0) parts.push(`tracks like ${sTracks.slice(0, 2).join(', ')}`);
  if (sArtists.length > 0) parts.push(`artists like ${sArtists.slice(0, 2).join(', ')}`);

  if (parts.length > 0) {
    const extra = trusted.find((d) => d.factor !== 'taste');
    return extra
      ? `Shared ${parts.join(' and ')}, plus strong alignment in ${extra.label}`
      : `You both connect over ${parts.join(' and ')}`;
  }

  if (trusted.length === 0) {
    return 'Early signal — still learning from both profiles';
  }
  const topLabels = trusted.slice(0, 2).map((d) => d.label);
  return topLabels.length === 2
    ? `Strong compatibility in ${topLabels[0]} and ${topLabels[1]}`
    : `Strong compatibility in ${topLabels[0]}`;
}

export function compatibilityScore(
  viewer: UserProfile,
  candidate: UserProfile,
  behaviorStats: Record<string, UserBehaviorStats>,
  weights: Record<LayerName, number>,
  mu: Record<LayerName, number>,
  clusterQuality = 1.0
): CompatibilityResult {
  const fwd = directionalScore(viewer, candidate, behaviorStats, weights, mu);
  const rev = directionalScore(candidate, viewer, behaviorStats, weights, mu);

  const reciprocal = harmonicMean(fwd.score, rev.score);
  const degraded = reciprocal * (0.85 + 0.15 * Math.max(0, Math.min(1, clusterQuality)));
  const drivers = buildMatchDrivers(fwd.details, weights);
  const sA = sharedArtists(viewer, candidate);
  const sT = sharedTracks(viewer, candidate);

  return {
    total: Number(degraded.toFixed(4)),
    forward: Number(fwd.score.toFixed(4)),
    reverse: Number(rev.score.toFixed(4)),
    layersForward: fwd.details,
    layersReverse: rev.details,
    weights,
    drivers,
    reason: buildMatchReason(sA, sT, drivers),
    clusterQuality: Number(clusterQuality.toFixed(3)),
  };
}

// Baseline legacy scorer reproducing v1 constants from the repo for A/B comparison
export function baselineCompatibilityScore(viewer: UserProfile, candidate: UserProfile): number {
  const music = Math.max(0, cosineSimilarity(viewer.rawTasteVec, candidate.rawTasteVec));
  const bio = Math.max(0, cosineSimilarity(viewer.rawPersonaVec, candidate.rawPersonaVec));
  const identity = 0.7 * music + 0.3 * bio;
  const pref = 0.5;
  const behavior = 0.5;
  return Number((0.6 * identity + 0.25 * pref + 0.15 * behavior).toFixed(4));
}
