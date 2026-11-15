import { cosineSimilarity, EMBED_DIM, normalizeVector } from './embeddings';
import {
  ClusteringResult,
  GameTheoryConfig,
  SetType,
  SwipeEvent,
  UserBehaviorStats,
  UserProfile,
  UserSet,
} from './types';

function euclideanDistance(a: number[], b: number[]): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    sum += d * d;
  }
  return Math.sqrt(sum);
}

function profileRichness(u: UserProfile): number {
  let score = 0;
  if (u.artists.length >= 3) score += 0.25;
  if (u.tracks.length >= 4) score += 0.25;
  if (u.musicMoods.length + u.musicEras.length + u.musicContexts.length >= 2) score += 0.2;
  if (u.bio.trim().length >= 25) score += 0.2;
  if (u.city.trim() && u.intent) score += 0.1;
  return Math.min(1, score);
}

// Simple DBSCAN over Attraction Signatures (OPEN DECISION 3 in MATCHING FORMULA.md)
function runDbscan(
  items: { userId: string; vec: number[] }[],
  eps: number,
  minSamples: number
): number[] {
  const n = items.length;
  const labels = new Array<number>(n).fill(-1); // -1 = noise/unassigned
  const visited = new Array<boolean>(n).fill(false);
  let clusterId = 0;

  const regionQuery = (idx: number): number[] => {
    const neighbors: number[] = [];
    for (let j = 0; j < n; j++) {
      if (euclideanDistance(items[idx].vec, items[j].vec) <= eps) {
        neighbors.push(j);
      }
    }
    return neighbors;
  };

  for (let i = 0; i < n; i++) {
    if (visited[i]) continue;
    visited[i] = true;
    const neighbors = regionQuery(i);
    if (neighbors.length < minSamples) continue;

    labels[i] = clusterId;
    const queue = [...neighbors];
    let qIdx = 0;
    while (qIdx < queue.length) {
      const curr = queue[qIdx++];
      if (!visited[curr]) {
        visited[curr] = true;
        const currNeighbors = regionQuery(curr);
        if (currNeighbors.length >= minSamples) {
          for (const nb of currNeighbors) {
            if (!queue.includes(nb)) queue.push(nb);
          }
        }
      }
      if (labels[curr] === -1) {
        labels[curr] = clusterId;
      }
    }
    clusterId += 1;
  }

  return labels;
}

export function buildMultiSetClusters(
  users: UserProfile[],
  swipes: SwipeEvent[],
  behaviorStats: Record<string, UserBehaviorStats>,
  config: GameTheoryConfig
): ClusteringResult {
  const sets: UserSet[] = [];
  const userMap = new Map(users.map((u) => [u.id, u]));

  // Precompute mutual match rate per user for quality gates
  const directedPositive = new Set<string>();
  for (const s of swipes) {
    if (s.action === 'LIKE' || s.action === 'SUPER_LIKE') {
      directedPositive.add(`${s.swiperId}::${s.targetId}`);
    }
  }
  const userMutualCount = new Map<string, number>();
  const userOutCount = new Map<string, number>();
  for (const s of swipes) {
    userOutCount.set(s.swiperId, (userOutCount.get(s.swiperId) || 0) + 1);
    if (
      (s.action === 'LIKE' || s.action === 'SUPER_LIKE') &&
      directedPositive.has(`${s.targetId}::${s.swiperId}`)
    ) {
      userMutualCount.set(s.swiperId, (userMutualCount.get(s.swiperId) || 0) + 1);
    }
  }

  const evaluateSetHealth = (set: UserSet) => {
    const members = set.memberIds.map((id) => userMap.get(id)).filter(Boolean) as UserProfile[];
    set.size = members.length;
    if (members.length === 0) {
      set.healthy = false;
      set.failureReasons = ['empty'];
      return;
    }
    const richness = members.reduce((s, u) => s + profileRichness(u), 0) / members.length;
    const activeSwipers = members.filter((u) => (userOutCount.get(u.id) || 0) > 0);
    const matchRate =
      activeSwipers.length > 0
        ? activeSwipers.reduce(
            (s, u) => s + (userMutualCount.get(u.id) || 0) / Math.max(1, userOutCount.get(u.id) || 1),
            0
          ) / activeSwipers.length
        : 0.25; // cold-start grace before swipes exist
    const minInactive = Math.min(...members.map((u) => u.daysSinceActive));

    set.avgRichness = Number(richness.toFixed(3));
    set.matchSuccessRate = Number(matchRate.toFixed(3));
    set.daysSinceActive = minInactive;

    const reasons: string[] = [];
    if (set.size < 10) reasons.push('size < 10');
    if (set.avgRichness < 0.3) reasons.push('richness < 0.30');
    if (activeSwipers.length >= 5 && set.matchSuccessRate < 0.1) reasons.push('match_rate < 0.10');
    if (set.daysSinceActive > 90) reasons.push('stale > 90d');

    set.failureReasons = reasons;
    set.healthy = reasons.length === 0;
  };

  // 1. Demographic Sets (Set-D): City + Age Cohort
  const demoBuckets = new Map<string, string[]>();
  for (const u of users) {
    const cohort = u.age < 25 ? '18-24' : u.age < 31 ? '25-30' : '31+';
    const key = `${u.city}::${cohort}`;
    const list = demoBuckets.get(key);
    if (list) list.push(u.id);
    else demoBuckets.set(key, [u.id]);
  }
  for (const [key, memberIds] of demoBuckets.entries()) {
    const [city, cohort] = key.split('::');
    const weights: Record<string, number> = {};
    for (const id of memberIds) weights[id] = 1.0;
    const s: UserSet = {
      id: `set-d-${key.toLowerCase().replace(/[^a-z0-9]+/g, '-')}`,
      type: 'demographic',
      label: `${city} (${cohort})`,
      memberIds,
      memberWeights: weights,
      centroid: null,
      size: memberIds.length,
      avgRichness: 0,
      matchSuccessRate: 0,
      daysSinceActive: 0,
      healthy: true,
      failureReasons: [],
      mergedIntoId: null,
    };
    evaluateSetHealth(s);
    sets.push(s);
  }

  // 2. Cold-Start Taste Sets (Set-M): Greedy spherical centroids with soft top-2 membership
  const tasteSeeds: { label: string; centroid: number[]; ids: string[] }[] = [];
  for (const u of users) {
    if (u.tasteVec.length !== EMBED_DIM) continue;
    let bestIdx = -1;
    let bestSim = -1;
    for (let i = 0; i < tasteSeeds.length; i++) {
      const sim = cosineSimilarity(u.tasteVec, tasteSeeds[i].centroid);
      if (sim > bestSim) {
        bestSim = sim;
        bestIdx = i;
      }
    }
    if (bestIdx === -1 || (bestSim < 0.36 && tasteSeeds.length < 6)) {
      const primaryTag = u.artists[0]?.tags[0] || u.artists[0]?.name || `Vibe ${tasteSeeds.length + 1}`;
      tasteSeeds.push({ label: `${primaryTag} taste`, centroid: [...u.tasteVec], ids: [u.id] });
    } else {
      const seed = tasteSeeds[bestIdx];
      seed.ids.push(u.id);
      const count = seed.ids.length;
      seed.centroid = normalizeVector(
        seed.centroid.map((v, k) => (v * (count - 1) + u.tasteVec[k]) / count)
      );
    }
  }

  // Soft multi-membership: each user joins up to top-2 taste sets weighted by (1 - lambda_behavior)
  const tasteMemberMaps: Record<string, Record<string, number>> = {};
  for (let i = 0; i < tasteSeeds.length; i++) tasteMemberMaps[i] = {};
  for (const u of users) {
    if (u.tasteVec.length !== EMBED_DIM) continue;
    const ranked = tasteSeeds
      .map((seed, idx) => ({ idx, sim: Math.max(0, cosineSimilarity(u.tasteVec, seed.centroid)) }))
      .sort((a, b) => b.sim - a.sim)
      .slice(0, 2);
    const lamB = behaviorStats[u.id]?.lambdaBehavior ?? 0;
    const coldWeight = 1 - 0.6 * lamB;
    for (const r of ranked) {
      if (r.sim > 0.15) {
        tasteMemberMaps[r.idx][u.id] = Number((r.sim * coldWeight).toFixed(3));
      }
    }
  }
  for (let i = 0; i < tasteSeeds.length; i++) {
    const memberWeights = tasteMemberMaps[i];
    const memberIds = Object.keys(memberWeights);
    const s: UserSet = {
      id: `set-m-${i + 1}`,
      type: 'taste',
      label: tasteSeeds[i].label,
      memberIds,
      memberWeights,
      centroid: tasteSeeds[i].centroid,
      size: memberIds.length,
      avgRichness: 0,
      matchSuccessRate: 0,
      daysSinceActive: 0,
      healthy: true,
      failureReasons: [],
      mergedIntoId: null,
    };
    evaluateSetHealth(s);
    sets.push(s);
  }

  // 3. Behavioral Engagement Sets (Set-B)
  const behaviorBuckets: Record<string, string[]> = {
    selective_active: [],
    exploratory_new: [],
    high_volume_damped: [],
  };
  for (const u of users) {
    const st = behaviorStats[u.id];
    if (!st || st.totalSwipes < 15) behaviorBuckets.exploratory_new.push(u.id);
    else if (st.rightSwipeRate > 0.65) behaviorBuckets.high_volume_damped.push(u.id);
    else behaviorBuckets.selective_active.push(u.id);
  }
  const behaviorLabels: Record<string, string> = {
    selective_active: 'Selective & engaged',
    exploratory_new: 'Cold-start / exploratory',
    high_volume_damped: 'High-volume swipers',
  };
  for (const [key, memberIds] of Object.entries(behaviorBuckets)) {
    const memberWeights: Record<string, number> = {};
    for (const id of memberIds) memberWeights[id] = 1.0;
    const s: UserSet = {
      id: `set-b-${key}`,
      type: 'behavior',
      label: behaviorLabels[key],
      memberIds,
      memberWeights,
      centroid: null,
      size: memberIds.length,
      avgRichness: 0,
      matchSuccessRate: 0,
      daysSinceActive: 0,
      healthy: true,
      failureReasons: [],
      mergedIntoId: null,
    };
    evaluateSetHealth(s);
    sets.push(s);
  }

  // 4. Attraction-Profile Clusters (Set-P) via DBSCAN on active users' attraction signatures
  const activeAttractions = users
    .filter((u) => behaviorStats[u.id]?.attractionSignature !== null)
    .map((u) => ({ userId: u.id, vec: behaviorStats[u.id].attractionSignature! }));

  let attractionClustersCount = 0;
  if (activeAttractions.length >= config.dbscanMinSamples) {
    const labels = runDbscan(activeAttractions, config.dbscanEps, config.dbscanMinSamples);
    const grouped = new Map<number, string[]>();
    labels.forEach((lbl, idx) => {
      if (lbl >= 0) {
        const list = grouped.get(lbl);
        if (list) list.push(activeAttractions[idx].userId);
        else grouped.set(lbl, [activeAttractions[idx].userId]);
      }
    });

    for (const [cid, coreIds] of grouped.entries()) {
      attractionClustersCount += 1;
      const centroidAcc = new Array<number>(EMBED_DIM).fill(0);
      for (const uid of coreIds) {
        const sig = behaviorStats[uid].attractionSignature!;
        for (let k = 0; k < EMBED_DIM; k++) centroidAcc[k] += sig[k];
      }
      const centroid = normalizeVector(centroidAcc);
      const memberWeights: Record<string, number> = {};
      for (const uid of coreIds) {
        const lam = behaviorStats[uid].lambdaBehavior;
        memberWeights[uid] = Number((0.4 + 0.6 * lam).toFixed(3));
      }
      const sampleUser = userMap.get(coreIds[0]);
      const tagHint = sampleUser?.artists[0]?.tags[0] || 'mixed';
      const s: UserSet = {
        id: `set-p-${cid + 1}`,
        type: 'attraction',
        label: `Attracted to ${tagHint} cluster #${cid + 1}`,
        memberIds: coreIds,
        memberWeights,
        centroid,
        size: coreIds.length,
        avgRichness: 0,
        matchSuccessRate: 0,
        daysSinceActive: 0,
        healthy: true,
        failureReasons: [],
        mergedIntoId: null,
      };
      evaluateSetHealth(s);
      sets.push(s);
    }
  }

  // Quality Gate Resolution: merge failed sets with centroids into nearest healthy set of same type
  let prunedSetsCount = 0;
  for (const s of sets) {
    if (s.healthy) continue;
    prunedSetsCount += 1;
    if (s.centroid) {
      let bestTarget: UserSet | null = null;
      let bestSim = -1;
      for (const cand of sets) {
        if (!cand.healthy || cand.id === s.id || !cand.centroid || cand.type !== s.type) continue;
        const sim = cosineSimilarity(s.centroid, cand.centroid);
        if (sim > bestSim) {
          bestSim = sim;
          bestTarget = cand;
        }
      }
      if (bestTarget) {
        s.mergedIntoId = bestTarget.id;
      }
    }
  }

  // Build user -> set membership index + per-user cluster quality scalar in [0, 1]
  const userSets: Record<string, { setId: string; type: SetType; weight: number; label: string }[]> = {};
  const userClusterQuality: Record<string, number> = {};
  for (const u of users) {
    userSets[u.id] = [];
  }

  for (const s of sets) {
    if (!s.healthy && !s.mergedIntoId) continue;
    const effectiveId = s.healthy ? s.id : s.mergedIntoId!;
    const targetSet = sets.find((x) => x.id === effectiveId) || s;
    for (const uid of s.memberIds) {
      userSets[uid].push({
        setId: targetSet.id,
        type: targetSet.type,
        weight: s.memberWeights[uid] ?? 1.0,
        label: targetSet.label,
      });
    }
  }

  for (const u of users) {
    const memberships = userSets[u.id];
    if (memberships.length === 0) {
      userClusterQuality[u.id] = 0.25;
      continue;
    }
    let qSum = 0;
    let wSum = 0;
    for (const m of memberships) {
      const setObj = sets.find((s) => s.id === m.setId);
      const setScore = setObj && setObj.healthy ? Math.min(1, setObj.avgRichness * 0.6 + setObj.matchSuccessRate * 0.4 + 0.2) : 0.4;
      qSum += setScore * m.weight;
      wSum += m.weight;
    }
    userClusterQuality[u.id] = Number(Math.min(1, wSum > 0 ? qSum / wSum : 0.5).toFixed(3));
  }

  return {
    sets,
    userSets,
    userClusterQuality,
    attractionClustersCount,
    prunedSetsCount,
  };
}
