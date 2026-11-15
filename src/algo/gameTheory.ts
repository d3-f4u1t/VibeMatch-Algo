import { cosineSimilarity } from './embeddings';
import {
  baselineCompatibilityScore,
  compatibilityScore,
  passesHardFilters,
  sharedArtists,
  sharedTracks,
} from './scoring';
import {
  ClusteringResult,
  CompatibilityResult,
  ConversationRecord,
  GameTheoryConfig,
  LayerName,
  QueueCandidate,
  SwipeEvent,
  UserBehaviorStats,
  UserProfile,
} from './types';

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-Math.max(-20, Math.min(20, x))));
}

export function computePerSetElo(
  users: UserProfile[],
  swipes: SwipeEvent[],
  conversations: ConversationRecord[],
  clustering: ClusteringResult,
  config: GameTheoryConfig
): Record<string, number> {
  const elo: Record<string, number> = {};
  for (const u of users) elo[u.id] = 1500;

  const directed = new Map<string, SwipeEvent>();
  for (const s of swipes) directed.set(`${s.swiperId}::${s.targetId}`, s);

  const convoSet = new Map<string, number>();
  for (const c of conversations) {
    const key = c.userA < c.userB ? `${c.userA}::${c.userB}` : `${c.userB}::${c.userA}`;
    convoSet.set(key, c.speakers.length);
  }

  // Primary set per user (prefer attraction -> taste -> demographic)
  const primarySet = (uid: string): string => {
    const list = clustering.userSets[uid] || [];
    const pref =
      list.find((x) => x.type === 'attraction') ||
      list.find((x) => x.type === 'taste') ||
      list[0];
    return pref ? pref.setId : 'global';
  };

  const settledCount: Record<string, number> = {};
  for (const s of swipes) {
    const isPos = s.action === 'LIKE' || s.action === 'SUPER_LIKE';
    if (!isPos) continue;
    const rev = directed.get(`${s.targetId}::${s.swiperId}`);
    const revPos = rev ? rev.action === 'LIKE' || rev.action === 'SUPER_LIKE' : false;
    const settled = s.ageDays >= config.settlingDays || rev !== undefined;
    if (!settled) continue;

    // Compare strictly within shared set context (or damped cross-set)
    const sameSet = primarySet(s.swiperId) === primarySet(s.targetId);
    const kBase = sameSet ? 24 : 12;

    const key = s.swiperId < s.targetId ? `${s.swiperId}::${s.targetId}` : `${s.targetId}::${s.swiperId}`;
    const speakers = convoSet.get(key) || 0;

    let actualForTarget = 0;
    if (speakers >= 2) actualForTarget = 1.0;
    else if (revPos) actualForTarget = 0.7;
    else actualForTarget = 0.0;

    const rSwiper = elo[s.swiperId] ?? 1500;
    const rTarget = elo[s.targetId] ?? 1500;
    const expectedTarget = 1 / (1 + Math.pow(10, (rSwiper - rTarget) / 400));

    settledCount[s.targetId] = (settledCount[s.targetId] || 0) + 1;
    const kFactor = settledCount[s.targetId] < 5 ? kBase * 0.5 : kBase;
    elo[s.targetId] = Number((rTarget + kFactor * (actualForTarget - expectedTarget)).toFixed(1));
  }

  return elo;
}

export function userExposureBudget(user: UserProfile, config: GameTheoryConfig): number {
  return Math.max(2, config.exposureBudgetBase - Math.floor(user.activeConversations / 2));
}

function expectedAcceptance(
  reverseCompat: number,
  proposerElo: number,
  receiverElo: number,
  receiverSelectivity: number,
  eloSensitivity: number
): number {
  const eloGap = Math.max(0, receiverElo - proposerElo) / 400;
  const logit = 3.2 * (reverseCompat - 0.45) - eloSensitivity * eloGap + 0.6 * (receiverSelectivity - 0.35);
  return sigmoid(logit);
}

interface RankedEdge {
  targetId: string;
  utility: number;
  compat: CompatibilityResult;
  expectedAccept: number;
}

function applyDiversityOrder(
  items: RankedEdge[],
  userMap: Map<string, UserProfile>,
  diversityPenalty: number
): RankedEdge[] {
  if (items.length <= 2 || diversityPenalty <= 0) return items;
  const remaining = [...items];
  const ordered: RankedEdge[] = [remaining.shift()!];

  while (remaining.length > 0) {
    const recentVecs = ordered
      .slice(-4)
      .map((x) => userMap.get(x.targetId)?.tasteVec || [])
      .filter((v) => v.length > 0);

    let bestIdx = 0;
    let bestScore = -Infinity;
    for (let i = 0; i < remaining.length; i++) {
      const cand = remaining[i];
      const vec = userMap.get(cand.targetId)?.tasteVec || [];
      let maxSim = 0;
      for (const rv of recentVecs) {
        const s = cosineSimilarity(vec, rv);
        if (s > maxSim) maxSim = s;
      }
      const adj = cand.utility - diversityPenalty * Math.max(0, maxSim);
      if (adj > bestScore) {
        bestScore = adj;
        bestIdx = i;
      }
    }
    ordered.push(remaining.splice(bestIdx, 1)[0]);
  }
  return ordered;
}

export function runBatchedStableMatching(
  users: UserProfile[],
  swipes: SwipeEvent[],
  behaviorStats: Record<string, UserBehaviorStats>,
  clustering: ClusteringResult,
  weights: Record<LayerName, number>,
  populationMeans: Record<LayerName, number>,
  eloByUserSet: Record<string, number>,
  config: GameTheoryConfig
): {
  queues: Record<string, QueueCandidate[]>;
  baselineTopMatches: Record<string, QueueCandidate[]>;
  exposureAssigned: Record<string, number>;
  exposureBaseline: Record<string, number>;
  bipartiteBlockingPairs: number;
  generalBlockingPairs: number;
} {
  const userMap = new Map(users.map((u) => [u.id, u]));
  const alreadySwiped = new Set(swipes.map((s) => `${s.swiperId}::${s.targetId}`));

  // Build candidate pool per user from overlapping multi-set union (fallback to global if small)
  const setMembersMap = new Map<string, Set<string>>();
  for (const s of clustering.sets) {
    if (!s.healthy && !s.mergedIntoId) continue;
    const targetId = s.healthy ? s.id : s.mergedIntoId!;
    let set = setMembersMap.get(targetId);
    if (!set) {
      set = new Set<string>();
      setMembersMap.set(targetId, set);
    }
    for (const uid of s.memberIds) set.add(uid);
  }

  const prefLists = new Map<string, RankedEdge[]>();
  const edgeLookup = new Map<string, RankedEdge>();
  const baselineTopMatches: Record<string, QueueCandidate[]> = {};
  const exposureBaseline: Record<string, number> = {};
  for (const u of users) exposureBaseline[u.id] = 0;

  for (const viewer of users) {
    // Union of active sets; ensure all same-city eligible users can be reached if pool is small
    const unionIds = new Set<string>();
    const memberships = clustering.userSets[viewer.id] || [];
    for (const m of memberships) {
      const members = setMembersMap.get(m.setId);
      if (members) for (const id of members) unionIds.add(id);
    }
    if (unionIds.size < 25) {
      for (const u of users) unionIds.add(u.id);
    }

    const viewerStats = behaviorStats[viewer.id];
    const viewerDamp = viewerStats?.selectivityDamping ?? 1.0;

    const scored: RankedEdge[] = [];
    const baseScored: { cand: UserProfile; sim: number }[] = [];

    for (const candId of unionIds) {
      const cand = userMap.get(candId);
      if (!cand) continue;
      if (alreadySwiped.has(`${viewer.id}::${cand.id}`)) continue;
      if (!passesHardFilters(viewer, cand)) continue;

      const candStats = behaviorStats[cand.id];
      const cq =
        ((clustering.userClusterQuality[viewer.id] ?? 0.7) +
          (clustering.userClusterQuality[cand.id] ?? 0.7)) /
        2;
      const compat = compatibilityScore(
        viewer,
        cand,
        behaviorStats,
        weights,
        populationMeans,
        cq
      );

      const pAccept = expectedAcceptance(
        compat.reverse,
        eloByUserSet[viewer.id] ?? 1500,
        eloByUserSet[cand.id] ?? 1500,
        candStats?.rightSwipeRate ?? 0.35,
        config.eloSensitivity
      );

      const candDamp = candStats?.selectivityDamping ?? 1.0;
      const utility = compat.total * pAccept * (0.1 + 0.9 * candDamp) * (0.1 + 0.9 * viewerDamp);

      const edge: RankedEdge = {
        targetId: cand.id,
        utility: Number(utility.toFixed(6)),
        compat,
        expectedAccept: Number(pAccept.toFixed(4)),
      };
      scored.push(edge);
      edgeLookup.set(`${viewer.id}::${cand.id}`, edge);

      baseScored.push({ cand, sim: baselineCompatibilityScore(viewer, cand) });
    }

    // Strict deterministic tie-breaking by targetId
    scored.sort((a, b) => b.utility - a.utility || a.targetId.localeCompare(b.targetId));
    prefLists.set(viewer.id, scored.slice(0, config.candidateTopM));

    // Legacy baseline top-K greedy queue
    baseScored.sort((a, b) => b.sim - a.sim || a.cand.id.localeCompare(b.cand.id));
    const baseTop = baseScored.slice(0, config.queueSizeK);
    baselineTopMatches[viewer.id] = baseTop.map((item, idx) => {
      exposureBaseline[item.cand.id] = (exposureBaseline[item.cand.id] || 0) + 1;
      return {
        userId: item.cand.id,
        compatibility: item.sim,
        expectedAcceptance: 0.5,
        utility: item.sim,
        rank: idx + 1,
        source: 'fallback',
        reason: 'Legacy greedy similarity',
        drivers: [],
        sharedArtists: sharedArtists(viewer, item.cand),
        sharedTracks: sharedTracks(viewer, item.cand),
      };
    });
  }

  // Partition users into clean Bipartite sub-market (straight men <-> straight women)
  // vs General-Graph orientation pool (bi/pan/gay/nonbinary).
  const isStraightMan = (u: UserProfile) =>
    u.gender === 'man' && u.interestedIn.length === 1 && u.interestedIn[0] === 'woman';
  const isStraightWoman = (u: UserProfile) =>
    u.gender === 'woman' && u.interestedIn.length === 1 && u.interestedIn[0] === 'man';

  const proposersBipartite = users.filter(isStraightMan);
  const receiversBipartite = users.filter(isStraightWoman);
  const bipartiteIdSet = new Set([
    ...proposersBipartite.map((u) => u.id),
    ...receiversBipartite.map((u) => u.id),
  ]);
  const generalUsers = users.filter((u) => !bipartiteIdSet.has(u.id));

  // Track mutual assignments: heldMatches[u] = Set of partnerIds
  const heldMatches = new Map<string, Set<string>>();
  for (const u of users) heldMatches.set(u.id, new Set<string>());

  // 1. Exact Many-to-Many Deferred Acceptance on Bipartite Sub-Market
  const qOut = config.queueSizeK;
  const receiverCap = new Map<string, number>();
  for (const r of receiversBipartite) {
    receiverCap.set(r.id, Math.min(config.queueSizeK, userExposureBudget(r, config)));
  }

  // Filtered preference lists restricted to the bipartite graph
  const biPrefs = new Map<string, RankedEdge[]>();
  for (const p of proposersBipartite) {
    biPrefs.set(
      p.id,
      (prefLists.get(p.id) || []).filter((e) => {
        const t = userMap.get(e.targetId);
        return t ? isStraightWoman(t) : false;
      })
    );
  }

  const nextIdx = new Map<string, number>();
  for (const p of proposersBipartite) nextIdx.set(p.id, 0);

  const receiverScore = (rId: string, pId: string): number => {
    const e = edgeLookup.get(`${rId}::${pId}`);
    if (!e) return -1;
    const heldCount = heldMatches.get(rId)?.size || 0;
    const budget = receiverCap.get(rId) || config.queueSizeK;
    const congestion = 1 / (1 + config.congestionPenalty * Math.pow(heldCount / Math.max(1, budget), 2));
    return e.utility * congestion;
  };

  // Deferred Acceptance loop
  let active = true;
  let guard = 0;
  while (active && guard < users.length * config.candidateTopM * 2) {
    active = false;
    guard += 1;
    for (const p of proposersBipartite) {
      const pHeld = heldMatches.get(p.id)!;
      const pList = biPrefs.get(p.id) || [];
      let idx = nextIdx.get(p.id) || 0;

      while (pHeld.size < qOut && idx < pList.length) {
        active = true;
        const targetId = pList[idx].targetId;
        idx += 1;
        nextIdx.set(p.id, idx);

        if (pHeld.has(targetId)) continue;
        // Ensure target actually finds proposer acceptable
        if (!edgeLookup.has(`${targetId}::${p.id}`)) continue;

        const rHeld = heldMatches.get(targetId)!;
        const cap = receiverCap.get(targetId) || qOut;

        if (rHeld.size < cap) {
          rHeld.add(p.id);
          pHeld.add(targetId);
        } else {
          // Find weakest currently held partner of receiver
          let worstPartner: string | null = null;
          let worstScore = Infinity;
          for (const existing of rHeld) {
            const sc = receiverScore(targetId, existing);
            if (sc < worstScore || (sc === worstScore && worstPartner && existing > worstPartner)) {
              worstScore = sc;
              worstPartner = existing;
            }
          }
          const candScore = receiverScore(targetId, p.id);
          if (
            worstPartner &&
            (candScore > worstScore || (candScore === worstScore && p.id < worstPartner))
          ) {
            rHeld.delete(worstPartner);
            heldMatches.get(worstPartner)?.delete(targetId);
            rHeld.add(p.id);
            pHeld.add(targetId);
          }
        }
      }
    }
  }

  // Count bipartite blocking pairs (should be strictly 0!)
  let bipartiteBlockingPairs = 0;
  for (const p of proposersBipartite) {
    const pHeld = heldMatches.get(p.id)!;
    const pList = biPrefs.get(p.id) || [];
    let pWorst = -Infinity;
    if (pHeld.size >= qOut) {
      pWorst = Math.min(...Array.from(pHeld).map((rid) => edgeLookup.get(`${p.id}::${rid}`)?.utility ?? 0));
    }
    for (const edge of pList) {
      if (pHeld.has(edge.targetId)) continue;
      const revEdge = edgeLookup.get(`${edge.targetId}::${p.id}`);
      if (!revEdge) continue;
      const pPrefers = pHeld.size < qOut || edge.utility > pWorst;
      if (!pPrefers) continue;

      const rHeld = heldMatches.get(edge.targetId)!;
      const rCap = receiverCap.get(edge.targetId) || qOut;
      const rWorst =
        rHeld.size >= rCap
          ? Math.min(...Array.from(rHeld).map((pid) => edgeLookup.get(`${edge.targetId}::${pid}`)?.utility ?? 0))
          : -Infinity;
      if (rHeld.size < rCap || revEdge.utility > rWorst) {
        bipartiteBlockingPairs += 1;
      }
    }
  }

  // 2. Non-Bipartite General-Graph Capacity-K Stable Matching + Blocking-Pair Local Repair
  const generalIdSet = new Set(generalUsers.map((u) => u.id));
  const genPairs: { u: string; v: string; mutualUtil: number }[] = [];
  const seenGen = new Set<string>();

  for (const u of generalUsers) {
    for (const edge of prefLists.get(u.id) || []) {
      if (!generalIdSet.has(edge.targetId)) continue;
      const rev = edgeLookup.get(`${edge.targetId}::${u.id}`);
      if (!rev) continue;
      const key = u.id < edge.targetId ? `${u.id}::${edge.targetId}` : `${edge.targetId}::${u.id}`;
      if (seenGen.has(key)) continue;
      seenGen.add(key);
      // Mutual harmonic utility seeds Irving-style stable pairing
      const hUtil =
        edge.utility > 0 && rev.utility > 0
          ? (2 * edge.utility * rev.utility) / (edge.utility + rev.utility)
          : 0;
      genPairs.push({ u: u.id, v: edge.targetId, mutualUtil: hUtil });
    }
  }

  genPairs.sort((a, b) => b.mutualUtil - a.mutualUtil);
  const genCap = new Map<string, number>();
  for (const u of generalUsers) {
    genCap.set(u.id, Math.min(config.queueSizeK, userExposureBudget(u, config)));
  }

  for (const pair of genPairs) {
    const hu = heldMatches.get(pair.u)!;
    const hv = heldMatches.get(pair.v)!;
    if (hu.size < (genCap.get(pair.u) || qOut) && hv.size < (genCap.get(pair.v) || qOut)) {
      hu.add(pair.v);
      hv.add(pair.u);
    }
  }

  // Blocking-pair local repair for general pool
  for (let pass = 0; pass < 3; pass++) {
    let swapped = false;
    for (const pair of genPairs) {
      const hu = heldMatches.get(pair.u)!;
      const hv = heldMatches.get(pair.v)!;
      if (hu.has(pair.v)) continue;

      const uCap = genCap.get(pair.u) || qOut;
      const vCap = genCap.get(pair.v) || qOut;
      const uLikesV = edgeLookup.get(`${pair.u}::${pair.v}`)?.utility ?? 0;
      const vLikesU = edgeLookup.get(`${pair.v}::${pair.u}`)?.utility ?? 0;

      const findWorst = (owner: string, held: Set<string>): { id: string | null; util: number } => {
        let worstId: string | null = null;
        let worstUtil = Infinity;
        for (const p of held) {
          const val = edgeLookup.get(`${owner}::${p}`)?.utility ?? 0;
          if (val < worstUtil) {
            worstUtil = val;
            worstId = p;
          }
        }
        return { id: worstId, util: worstUtil };
      };

      const uWorst = hu.size >= uCap ? findWorst(pair.u, hu) : { id: null, util: -Infinity };
      const vWorst = hv.size >= vCap ? findWorst(pair.v, hv) : { id: null, util: -Infinity };

      if (uLikesV > uWorst.util && vLikesU > vWorst.util) {
        if (uWorst.id) {
          hu.delete(uWorst.id);
          heldMatches.get(uWorst.id)?.delete(pair.u);
        }
        if (vWorst.id) {
          hv.delete(vWorst.id);
          heldMatches.get(vWorst.id)?.delete(pair.v);
        }
        hu.add(pair.v);
        hv.add(pair.u);
        swapped = true;
      }
    }
    if (!swapped) break;
  }

  // Audit remaining general blocking pairs
  let generalBlockingPairs = 0;
  for (const pair of genPairs) {
    const hu = heldMatches.get(pair.u)!;
    const hv = heldMatches.get(pair.v)!;
    if (hu.has(pair.v)) continue;
    const uCap = genCap.get(pair.u) || qOut;
    const vCap = genCap.get(pair.v) || qOut;
    const uLikesV = edgeLookup.get(`${pair.u}::${pair.v}`)?.utility ?? 0;
    const vLikesU = edgeLookup.get(`${pair.v}::${pair.u}`)?.utility ?? 0;
    const uWorst =
      hu.size >= uCap
        ? Math.min(...Array.from(hu).map((p) => edgeLookup.get(`${pair.u}::${p}`)?.utility ?? 0))
        : -Infinity;
    const vWorst =
      hv.size >= vCap
        ? Math.min(...Array.from(hv).map((p) => edgeLookup.get(`${pair.v}::${p}`)?.utility ?? 0))
        : -Infinity;
    if (uLikesV > uWorst && vLikesU > vWorst) {
      generalBlockingPairs += 1;
    }
  }

  // Assemble final MMR-diverse queues + fallback fill up to K
  const queues: Record<string, QueueCandidate[]> = {};
  const exposureAssigned: Record<string, number> = {};
  for (const u of users) exposureAssigned[u.id] = 0;

  for (const u of users) {
    const isBi = bipartiteIdSet.has(u.id);
    const heldIds = Array.from(heldMatches.get(u.id) || []);
    const stableEdges = heldIds
      .map((tid) => edgeLookup.get(`${u.id}::${tid}`))
      .filter(Boolean) as RankedEdge[];

    stableEdges.sort((a, b) => b.utility - a.utility);
    const orderedStable = applyDiversityOrder(stableEdges, userMap, config.diversityPenalty);

    const queue: QueueCandidate[] = [];
    const used = new Set<string>();

    for (const e of orderedStable) {
      used.add(e.targetId);
      const target = userMap.get(e.targetId)!;
      exposureAssigned[e.targetId] = (exposureAssigned[e.targetId] || 0) + 1;
      queue.push({
        userId: e.targetId,
        compatibility: e.compat.total,
        expectedAcceptance: e.expectedAccept,
        utility: e.utility,
        rank: queue.length + 1,
        source: isBi ? 'stable_bipartite' : 'stable_general',
        reason: e.compat.reason,
        drivers: e.compat.drivers,
        sharedArtists: sharedArtists(u, target),
        sharedTracks: sharedTracks(u, target),
      });
    }

    // Real-time fallback if stable batch has < K cards: sort by congestion-discounted utility and enforce budget
    const fallbacks = (prefLists.get(u.id) || [])
      .filter((e) => !used.has(e.targetId))
      .map((e) => {
        const target = userMap.get(e.targetId)!;
        const budget = userExposureBudget(target, config);
        const load = exposureAssigned[e.targetId] || 0;
        const discounted = e.utility / (1 + config.congestionPenalty * Math.pow(load / Math.max(1, budget), 2));
        return { ...e, utility: discounted };
      })
      .sort((a, b) => b.utility - a.utility);
    const orderedFallback = applyDiversityOrder(fallbacks, userMap, config.diversityPenalty);
    for (const e of orderedFallback) {
      if (queue.length >= config.queueSizeK) break;
      const target = userMap.get(e.targetId)!;
      const budget = userExposureBudget(target, config);
      if ((exposureAssigned[e.targetId] || 0) >= budget) continue;
      exposureAssigned[e.targetId] = (exposureAssigned[e.targetId] || 0) + 1;
      queue.push({
        userId: e.targetId,
        compatibility: e.compat.total,
        expectedAcceptance: e.expectedAccept,
        utility: e.utility,
        rank: queue.length + 1,
        source: 'fallback',
        reason: e.compat.reason,
        drivers: e.compat.drivers,
        sharedArtists: sharedArtists(u, target),
        sharedTracks: sharedTracks(u, target),
      });
    }

    queues[u.id] = queue;
  }

  return {
    queues,
    baselineTopMatches,
    exposureAssigned,
    exposureBaseline,
    bipartiteBlockingPairs,
    generalBlockingPairs,
  };
}
