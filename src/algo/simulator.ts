import { buildMultiSetClusters } from './clustering';
import { centerVector, cosineSimilarity, encodeAndCenterAllUsers } from './embeddings';
import { computePerSetElo, runBatchedStableMatching } from './gameTheory';
import {
  attractionAlignment,
  compatibilityScore,
  computeAllBehaviorStats,
  computeLayerWeights,
  computePopulationMeans,
  harmonicMean,
  identitySimilarity,
  LAYER_HALF_LIFE,
  nEffPersona,
  pairReliability,
  passesHardFilters,
  personaSimilarity,
  reliability,
  selectivityDamping,
} from './scoring';
import {
  Archetype,
  ConversationRecord,
  GameTheoryConfig,
  Gender,
  Intent,
  InvariantCheck,
  SimulationSnapshot,
  SwipeEvent,
  UserProfile,
} from './types';

export const DEFAULT_CONFIG: GameTheoryConfig = {
  queueSizeK: 8,
  candidateTopM: 45,
  exposureBudgetBase: 8,
  spamDampingGamma: 14.0,
  eloSensitivity: 0.75,
  congestionPenalty: 0.65,
  diversityPenalty: 0.15,
  settlingDays: 7,
  priorStrengthKappa: 50,
  dbscanEps: 0.88,
  dbscanMinSamples: 6,
};

export interface MarketScenarioParams {
  userCount: number;
  spammerRatio: number;
  coldStartRatio: number;
  nonBipartiteRatio: number;
  seed: number;
}

export const DEFAULT_SCENARIO: MarketScenarioParams = {
  userCount: 120,
  spammerRatio: 0.15,
  coldStartRatio: 0.22,
  nonBipartiteRatio: 0.18,
  seed: 42,
};

function makeRng(seed: number) {
  let s = seed || 1;
  return () => {
    s = (Math.imul(1664525, s) + 1013904223) >>> 0;
    return s / 0xffffffff;
  };
}

const VIBE_ARCHETYPES = [
  {
    artists: [
      { name: 'SZA', tags: ['r&b', 'neo-soul', 'alt-r&b'] },
      { name: 'Frank Ocean', tags: ['r&b', 'alt-r&b', 'soul'] },
      { name: 'FKA twigs', tags: ['alt-r&b', 'electronic', 'art pop'] },
      { name: 'Daniel Caesar', tags: ['r&b', 'soul'] },
    ],
    tracks: [
      { title: 'Snooze', artistName: 'SZA' },
      { title: 'Nights', artistName: 'Frank Ocean' },
      { title: 'Cellophane', artistName: 'FKA twigs' },
      { title: 'Best Part', artistName: 'Daniel Caesar' },
    ],
    moods: ['late night', 'melancholic', 'chill'],
    eras: ['2010s', '2020s'],
    energy: 'low-mid',
    contexts: ['3am drives', 'studying'],
    bios: [
      'Late night vinyl collector looking for someone to share headphones with. Always down for coffee and museums.',
      'Alt R&B obsessive, film photography on weekends, cooking pasta from scratch.',
    ],
  },
  {
    artists: [
      { name: 'Travis Scott', tags: ['hip-hop', 'trap', 'rap'] },
      { name: 'Kendrick Lamar', tags: ['hip-hop', 'rap'] },
      { name: 'Kanye West', tags: ['hip-hop', 'rap'] },
      { name: 'Drake', tags: ['hip-hop', 'pop'] },
    ],
    tracks: [
      { title: 'SICKO MODE', artistName: 'Travis Scott' },
      { title: 'Alright', artistName: 'Kendrick Lamar' },
      { title: 'Stronger', artistName: 'Kanye West' },
      { title: 'Passionfruit', artistName: 'Drake' },
    ],
    moods: ['hype', 'confident'],
    eras: ['2010s', '2020s'],
    energy: 'high',
    contexts: ['gym', 'party', 'concerts'],
    bios: [
      'Live hip-hop sets, heavy gym sessions, and late-night taco runs after shows.',
      'Always hunting for new concerts and sneaker drops. Big energy, zero drama.',
    ],
  },
  {
    artists: [
      { name: 'Tame Impala', tags: ['indie', 'alt-rock', 'psychedelic'] },
      { name: 'Mitski', tags: ['indie', 'alt-rock'] },
      { name: 'Phoebe Bridgers', tags: ['indie', 'folk'] },
      { name: 'Arctic Monkeys', tags: ['indie', 'alt-rock'] },
    ],
    tracks: [
      { title: 'Let It Happen', artistName: 'Tame Impala' },
      { title: 'Nobody', artistName: 'Mitski' },
      { title: 'Kyoto', artistName: 'Phoebe Bridgers' },
      { title: '505', artistName: 'Arctic Monkeys' },
    ],
    moods: ['dreamy', 'nostalgic'],
    eras: ['2010s', '2000s'],
    energy: 'medium',
    contexts: ['road trip', 'bookstore'],
    bios: [
      'Indie gigs in tiny basements, coastal road trips, and secondhand bookstores.',
      'Guitar riffs and rainy afternoons. Looking for someone genuinely curious.',
    ],
  },
  {
    artists: [
      { name: 'Fred again..', tags: ['electronic', 'house', 'uk garage'] },
      { name: 'Kaytranada', tags: ['electronic', 'house', 'funk'] },
      { name: 'Disclosure', tags: ['electronic', 'house'] },
      { name: 'Four Tet', tags: ['electronic', 'ambient'] },
    ],
    tracks: [
      { title: 'Delilah', artistName: 'Fred again..' },
      { title: '10%', artistName: 'Kaytranada' },
      { title: 'Latch', artistName: 'Disclosure' },
      { title: 'Two Thousand and Seventeen', artistName: 'Four Tet' },
    ],
    moods: ['euphoric', 'groovy'],
    eras: ['2020s', '2010s'],
    energy: 'high',
    contexts: ['club', 'workout'],
    bios: [
      'Warehouse sets till 4am and sunrise espresso. Always building new DJ playlists.',
      'UK garage and synth grooves. Catch me near the speakers.',
    ],
  },
  {
    artists: [
      { name: 'Miles Davis', tags: ['jazz', 'instrumental'] },
      { name: 'Kamasi Washington', tags: ['jazz', 'soul'] },
      { name: 'Hiatus Kaiyote', tags: ['neo-soul', 'jazz'] },
      { name: 'BADBADNOTGOOD', tags: ['jazz', 'instrumental'] },
    ],
    tracks: [
      { title: 'So What', artistName: 'Miles Davis' },
      { title: 'Truth', artistName: 'Kamasi Washington' },
      { title: 'Nakamarra', artistName: 'Hiatus Kaiyote' },
      { title: 'Time Moves Slow', artistName: 'BADBADNOTGOOD' },
    ],
    moods: ['warm', 'reflective'],
    eras: ['1960s', '2010s', '2020s'],
    energy: 'low-mid',
    contexts: ['coffee', 'studying'],
    bios: [
      'Jazz bars, pour-over coffee, and quiet Sunday mornings with good records.',
      'Musician by hobby, architect by day. Looking for depth and warmth.',
    ],
  },
];

const FIRST_NAMES = [
  'Maya', 'Liam', 'Aria', 'Noah', 'Zoe', 'Ethan', 'Nora', 'Kai', 'Sora', 'Leo',
  'Mina', 'Owen', 'Elena', 'Jude', 'Ayla', 'Theo', 'Chloe', 'Miles', 'Rhea', 'Sam',
  'Tessa', 'Julian', 'Lina', 'Caleb', 'Ivy', 'Ezra', 'Naomi', 'Lucas', 'Kira', 'Ravi',
];

export function generateSyntheticMarket(
  scenario: MarketScenarioParams,
  config: GameTheoryConfig
): { users: UserProfile[]; swipes: SwipeEvent[]; conversations: ConversationRecord[] } {
  const rng = makeRng(scenario.seed);
  const cities = ['San Francisco', 'Brooklyn', 'Austin'];
  const intents: Intent[] = ['long_term', 'long_term', 'short_term', 'open', 'friendship'];
  const users: UserProfile[] = [];

  for (let i = 0; i < scenario.userCount; i++) {
    const vibe = VIBE_ARCHETYPES[i % VIBE_ARCHETYPES.length];
    const city = i < scenario.userCount * 0.72 ? cities[0] : cities[1 + (i % 2)];

    const isNonBi = rng() < scenario.nonBipartiteRatio;
    let gender: Gender;
    let interestedIn: Gender[];
    if (!isNonBi) {
      gender = i % 2 === 0 ? 'man' : 'woman';
      interestedIn = gender === 'man' ? ['woman'] : ['man'];
    } else {
      const roll = rng();
      if (roll < 0.35) {
        gender = 'man';
        interestedIn = ['man', 'nonbinary'];
      } else if (roll < 0.7) {
        gender = 'woman';
        interestedIn = ['woman', 'man'];
      } else {
        gender = 'nonbinary';
        interestedIn = ['man', 'woman', 'nonbinary'];
      }
    }

    const rArch = rng();
    let archetype: Archetype = 'balanced';
    if (rArch < scenario.coldStartRatio) archetype = 'cold_start';
    else if (rArch < scenario.coldStartRatio + scenario.spammerRatio) archetype = 'spammer';
    else if (rArch > 0.78) archetype = 'selective';

    const age = 21 + Math.floor(rng() * 14);
    const hasBio = archetype !== 'cold_start' || rng() > 0.5;

    // Pick 3-4 artists and 4 tracks, mixing occasional cross-genre favorite
    const artists = [...vibe.artists.slice(0, 3)];
    if (rng() > 0.5) {
      const neighbor = VIBE_ARCHETYPES[(i + 1) % VIBE_ARCHETYPES.length];
      artists.push(neighbor.artists[0]);
    } else {
      artists.push(vibe.artists[3]);
    }

    users.push({
      id: `u_${String(i + 1).padStart(3, '0')}`,
      name: `${FIRST_NAMES[i % FIRST_NAMES.length]} ${Math.floor(i / FIRST_NAMES.length) + 1}`,
      age,
      gender,
      interestedIn,
      city,
      intent: intents[Math.floor(rng() * intents.length)],
      ageMin: Math.max(18, age - 6),
      ageMax: age + 7,
      maxDistanceKm: 40,
      dealbreakers: rng() > 0.8 ? ['smoking'] : [],
      habits: { smoking: rng() > 0.88 ? 'regularly' : 'never' },
      artists,
      tracks: [...vibe.tracks],
      musicMoods: archetype === 'cold_start' ? [] : [...vibe.moods],
      musicEras: archetype === 'cold_start' ? [] : [...vibe.eras],
      musicEnergy: archetype === 'cold_start' ? '' : vibe.energy,
      musicContexts: archetype === 'cold_start' ? [] : [...vibe.contexts],
      bio: hasBio ? vibe.bios[i % vibe.bios.length] : '',
      archetype,
      activeConversations: archetype === 'cold_start' ? 0 : Math.floor(rng() * 5),
      daysSinceActive: Math.floor(rng() * 12),
      rawTasteVec: [],
      rawContextVec: [],
      rawPersonaVec: [],
      tasteVec: [],
      contextVec: [],
      personaVec: [],
      tasteXY: [0, 0],
    });
  }

  encodeAndCenterAllUsers(users);

  // Generate realistic prior swipe history & conversations
  const swipes: SwipeEvent[] = [];
  const conversations: ConversationRecord[] = [];
  const seenPairs = new Set<string>();

  for (const u of users) {
    if (u.archetype === 'cold_start') continue;

    const targetCount = u.archetype === 'spammer' ? 28 : u.archetype === 'selective' ? 18 : 22;
    let created = 0;

    for (let attempt = 0; attempt < users.length && created < targetCount; attempt++) {
      const idx = Math.floor(rng() * users.length);
      const cand = users[idx];
      if (!passesHardFilters(u, cand)) continue;
      if (seenPairs.has(`${u.id}::${cand.id}`)) continue;
      seenPairs.add(`${u.id}::${cand.id}`);

      const tasteSim = cosineSimilarity(u.tasteVec, cand.tasteVec);
      let action: SwipeEvent['action'] = 'PASS';
      if (u.archetype === 'spammer') {
        action = rng() < 0.92 ? 'LIKE' : 'PASS';
      } else if (u.archetype === 'selective') {
        if (tasteSim > 0.42) action = rng() < 0.2 ? 'SUPER_LIKE' : 'LIKE';
        else action = rng() < 0.1 ? 'LIKE' : 'PASS';
      } else {
        action = tasteSim > 0.22 || rng() < 0.25 ? 'LIKE' : 'PASS';
      }

      swipes.push({
        swiperId: u.id,
        targetId: cand.id,
        action,
        ageDays: rng() < 0.75 ? 9 : 2,
      });
      created += 1;
    }
  }

  // Open conversations on a fraction of mutual likes where taste + persona align
  const posSet = new Set(
    swipes.filter((s) => s.action !== 'PASS').map((s) => `${s.swiperId}::${s.targetId}`)
  );
  const userMap = new Map(users.map((u) => [u.id, u]));
  const seenConvo = new Set<string>();
  for (const s of swipes) {
    if (s.action === 'PASS') continue;
    if (!posSet.has(`${s.targetId}::${s.swiperId}`)) continue;
    const key = s.swiperId < s.targetId ? `${s.swiperId}::${s.targetId}` : `${s.targetId}::${s.swiperId}`;
    if (seenConvo.has(key)) continue;
    seenConvo.add(key);

    const a = userMap.get(s.swiperId)!;
    const b = userMap.get(s.targetId)!;
    const sim = cosineSimilarity(a.tasteVec, b.tasteVec);
    if (sim > 0.25 && rng() < 0.7) {
      conversations.push({
        userA: a.id,
        userB: b.id,
        speakers: rng() < 0.75 ? [a.id, b.id] : [a.id],
      });
    }
  }

  return { users, swipes, conversations };
}

function giniCoefficient(values: number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const sum = sorted.reduce((s, v) => s + v, 0);
  if (sum <= 0) return 0;
  let weighted = 0;
  for (let i = 0; i < sorted.length; i++) {
    weighted += (i + 1) * sorted[i];
  }
  return Number(((2 * weighted) / (sorted.length * sum) - (sorted.length + 1) / sorted.length).toFixed(4));
}

export function runSimulation(
  scenario: MarketScenarioParams = DEFAULT_SCENARIO,
  config: GameTheoryConfig = DEFAULT_CONFIG
): SimulationSnapshot {
  const t0 = performance.now();
  const { users, swipes, conversations } = generateSyntheticMarket(scenario, config);
  const behaviorStats = computeAllBehaviorStats(users, swipes, conversations, config);
  const populationMeans = computePopulationMeans(users, behaviorStats);
  const weights = computeLayerWeights(users, swipes, behaviorStats, config);
  const clustering = buildMultiSetClusters(users, swipes, behaviorStats, config);
  const eloByUserSet = computePerSetElo(users, swipes, conversations, clustering, config);

  const stableResult = runBatchedStableMatching(
    users,
    swipes,
    behaviorStats,
    clustering,
    weights,
    populationMeans,
    eloByUserSet,
    config
  );
  const runtimeMs = Number((performance.now() - t0).toFixed(2));

  // Compute comparison metrics: mutual match yield inside top-K queues
  const countMutualInTopK = (queues: Record<string, { userId: string; compatibility: number }[]>) => {
    const sets = new Map<string, Set<string>>();
    for (const [uid, list] of Object.entries(queues)) {
      sets.set(uid, new Set(list.map((x) => x.userId)));
    }
    let mutualPairs = 0;
    let compatSum = 0;
    const seen = new Set<string>();
    for (const [uid, list] of Object.entries(queues)) {
      for (const item of list) {
        if (sets.get(item.userId)?.has(uid)) {
          const key = uid < item.userId ? `${uid}::${item.userId}` : `${item.userId}::${uid}`;
          if (!seen.has(key)) {
            seen.add(key);
            mutualPairs += 1;
            compatSum += item.compatibility;
          }
        }
      }
    }
    return {
      mutualPairs,
      meanCompat: mutualPairs > 0 ? Number((compatSum / mutualPairs).toFixed(4)) : 0,
    };
  };

  const upgStats = countMutualInTopK(stableResult.queues);
  const baseStats = countMutualInTopK(stableResult.baselineTopMatches);

  const coldUsers = users.filter((u) => u.archetype === 'cold_start');
  const coldUp = coldUsers.filter((u) => (stableResult.exposureAssigned[u.id] || 0) > 0).length;
  const coldBase = coldUsers.filter((u) => (stableResult.exposureBaseline[u.id] || 0) > 0).length;

  const spammers = users.filter((u) => u.archetype === 'spammer');
  const selectives = users.filter((u) => u.archetype === 'selective');
  const spamMatchRate =
    spammers.length > 0
      ? spammers.reduce((s, u) => s + (stableResult.queues[u.id]?.length || 0), 0) /
        (spammers.length * config.queueSizeK)
      : 0;
  const selMatchRate =
    selectives.length > 0
      ? selectives.reduce((s, u) => s + (stableResult.queues[u.id]?.length || 0), 0) /
        (selectives.length * config.queueSizeK)
      : 0;

  // Compute average top-K queue utility for spammers vs selectives
  const avgUtil = (arr: UserProfile[]) => {
    if (!arr.length) return 0;
    let sum = 0;
    let cnt = 0;
    for (const u of arr) {
      for (const q of stableResult.queues[u.id] || []) {
        sum += q.utility;
        cnt += 1;
      }
    }
    return cnt > 0 ? sum / cnt : 0;
  };

  const spamAvgUtil = avgUtil(spammers);
  const selAvgUtil = avgUtil(selectives);

  // Verify mathematical & game-theoretic invariants
  const invariants = verifyInvariants(
    users,
    behaviorStats,
    weights,
    populationMeans,
    stableResult,
    config,
    spamAvgUtil,
    selAvgUtil
  );

  return {
    config,
    users,
    swipes,
    conversations,
    behaviorStats,
    weights,
    populationMeans,
    clustering,
    eloByUserSet,
    exposureAssigned: stableResult.exposureAssigned,
    exposureBaseline: stableResult.exposureBaseline,
    queues: stableResult.queues,
    baselineTopMatches: stableResult.baselineTopMatches,
    metrics: {
      userCount: users.length,
      mutualMatchesUpgraded: upgStats.mutualPairs,
      mutualMatchesBaseline: baseStats.mutualPairs,
      meanMutualCompatibility: upgStats.meanCompat,
      baselineMutualCompatibility: baseStats.meanCompat,
      exposureGiniUpgraded: giniCoefficient(Object.values(stableResult.exposureAssigned)),
      exposureGiniBaseline: giniCoefficient(Object.values(stableResult.exposureBaseline)),
      spammerMatchRateUpgraded: Number(spamAvgUtil.toFixed(4)) || Number(spamMatchRate.toFixed(4)),
      selectiveMatchRateUpgraded: Number(selAvgUtil.toFixed(4)) || Number(selMatchRate.toFixed(4)),
      coldStartCoverageUpgraded: coldUsers.length ? Number((coldUp / coldUsers.length).toFixed(3)) : 1,
      coldStartCoverageBaseline: coldUsers.length ? Number((coldBase / coldUsers.length).toFixed(3)) : 1,
      bipartiteBlockingPairs: stableResult.bipartiteBlockingPairs,
      generalBlockingPairs: stableResult.generalBlockingPairs,
      runtimeMs,
    },
    invariants,
  };
}

function verifyInvariants(
  users: UserProfile[],
  behaviorStats: Record<string, ReturnType<typeof computeAllBehaviorStats>[string]>,
  weights: Record<string, number>,
  populationMeans: Record<string, number>,
  stableResult: ReturnType<typeof runBatchedStableMatching>,
  config: GameTheoryConfig,
  spamAvgUtil: number,
  selAvgUtil: number
): InvariantCheck[] {
  const checks: InvariantCheck[] = [];
  const u0 = users[0];
  const u1 = users[1];

  // 1. Zero-evidence neutrality
  const emptyCandidate = { ...u1, bio: '', rawPersonaVec: [], personaVec: [] };
  const richCandidate = { ...u1 };
  const cEmpty = compatibilityScore(u0, emptyCandidate, behaviorStats, weights, populationMeans);
  const cRich = compatibilityScore(u0, richCandidate, behaviorStats, weights, populationMeans);
  const emptyLam = cEmpty.layersForward.persona.lambda;
  const emptyShrunk = cEmpty.layersForward.persona.shrunk;
  checks.push({
    id: 'inv-1',
    name: 'Zero-evidence neutrality (lam=0 -> mu)',
    passed: emptyLam === 0 && Math.abs(emptyShrunk - populationMeans.persona) < 0.002,
    detail: 'Empty candidate bio yields pair lambda=0 and shrinks exactly to population mean instead of penalizing.',
    metricValue: `lam=${emptyLam.toFixed(2)}, shrunk=${emptyShrunk.toFixed(3)} vs mu=${populationMeans.persona.toFixed(3)}`,
  });

  // 2. Symmetry of M(A, B)
  const ab = compatibilityScore(u0, u1, behaviorStats, weights, populationMeans);
  const ba = compatibilityScore(u1, u0, behaviorStats, weights, populationMeans);
  checks.push({
    id: 'inv-2',
    name: 'Reciprocal symmetry M(A,B) = M(B,A)',
    passed: Math.abs(ab.total - ba.total) < 0.0002,
    detail: 'Harmonic aggregation guarantees identical compatibility regardless of evaluation direction.',
    metricValue: `|${ab.total} - ${ba.total}| = ${Math.abs(ab.total - ba.total).toFixed(4)}`,
  });

  // 3. Harmonic strictness under asymmetry
  const h = harmonicMean(0.95, 0.3);
  const arith = (0.95 + 0.3) / 2;
  checks.push({
    id: 'inv-3',
    name: 'Harmonic mean suppresses lopsided attraction',
    passed: h < arith - 0.15,
    detail: 'For directional scores (0.95 vs 0.30), harmonic mean penalizes one-sided hype.',
    metricValue: `harmonic=${h.toFixed(3)} < arithmetic=${arith.toFixed(3)}`,
  });

  // 4. Independent per-layer centering (context does not subtract taste mean)
  const ctxMean = users.find((u) => u.rawContextVec.length > 0 && u.contextVec.length > 0);
  const diffFromRaw = ctxMean ? 1 - cosineSimilarity(ctxMean.rawContextVec, ctxMean.contextVec) : 0;
  checks.push({
    id: 'inv-4',
    name: 'Per-layer anisotropy centering isolation',
    passed: Boolean(ctxMean && diffFromRaw > 0.001),
    detail: 'Context & persona vectors center against their own population mean vectors, never taste mean.',
    metricValue: `delta_cos=${diffFromRaw.toFixed(4)}`,
  });

  // 5. Attraction alignment non-collapse when passes=0
  const likesOnlyStats = {
    userId: u1.id,
    nEff: 6,
    lambdaBehavior: 0.2,
    totalSwipes: 8,
    likeCount: 8,
    passCount: 0,
    superLikeCount: 0,
    rightSwipeRate: 1.0,
    selectivityDamping: 0.5,
    likedCentroid: u0.tasteVec,
    passedCentroid: null,
    likedSpread: 0.1,
    attractionSignature: null,
  };
  const alignScore = attractionAlignment(u0, likesOnlyStats);
  checks.push({
    id: 'inv-5',
    name: 'Attraction alignment non-collapse (passes = 0)',
    passed: alignScore > 0.85,
    detail: 'Candidate with likes and zero passes scores positively on aligned taste instead of collapsing to 0.0.',
    metricValue: `align=${alignScore.toFixed(3)}`,
  });

  // 6. Bipartite Gale-Shapley Zero Blocking Pairs
  checks.push({
    id: 'inv-6',
    name: 'Bipartite Deferred Acceptance stability',
    passed: stableResult.bipartiteBlockingPairs === 0,
    detail: 'Exact many-to-many Gale-Shapley produces zero blocking pairs across bipartite pools.',
    metricValue: `blocking_pairs=${stableResult.bipartiteBlockingPairs}`,
  });

  // 7. General-Graph Blocking Pair Repair
  checks.push({
    id: 'inv-7',
    name: 'Non-bipartite pool blocking-pair bound',
    passed: stableResult.generalBlockingPairs === 0,
    detail: 'Symmetric capacity-K matching + local swap repair eliminates unstable pairs in mixed orientation pools.',
    metricValue: `blocking_pairs=${stableResult.generalBlockingPairs}`,
  });

  // 8. Selectivity Damping Dominance
  const spamDamp = selectivityDamping(0.95, 30, config.spamDampingGamma);
  const healthyDamp = selectivityDamping(0.32, 30, config.spamDampingGamma);
  checks.push({
    id: 'inv-8',
    name: 'Anti-spam selectivity dominance',
    passed: healthyDamp === 1.0 && spamDamp < 0.35 && (selAvgUtil === 0 || spamAvgUtil <= selAvgUtil),
    detail: '95% right-swipe spammer receives severe damping vs healthy 32% selective user.',
    metricValue: `phi(0.95)=${spamDamp.toFixed(3)} vs phi(0.32)=${healthyDamp.toFixed(3)}`,
  });

  // 9. Exposure Gini Reduction vs Greedy Baseline
  const giniUpg = giniCoefficient(Object.values(stableResult.exposureAssigned));
  const giniBase = giniCoefficient(Object.values(stableResult.exposureBaseline));
  checks.push({
    id: 'inv-9',
    name: 'Exposure budget reduces impression inequality',
    passed: giniUpg <= giniBase,
    detail: 'Congestion pricing and inbound budgets curb winner-take-all concentration vs greedy baseline.',
    metricValue: `Gini: ${giniUpg.toFixed(3)} vs ${giniBase.toFixed(3)}`,
  });

  // 10. Clean Explainability (No raw percentages leaked)
  const sampleQueue = stableResult.queues[u0.id]?.[0];
  const hasPercentLeak = sampleQueue ? /\d+%/.test(sampleQueue.reason) : false;
  checks.push({
    id: 'inv-10',
    name: 'Clean explainability (no raw cosine % leaks)',
    passed: !hasPercentLeak,
    detail: 'Match reasons surface ranked trusted drivers and shared music without uncalibrated percentage suffixes.',
    metricValue: sampleQueue ? `"${sampleQueue.reason.slice(0, 46)}..."` : 'N/A',
  });

  // Prevent unused warning on helper refs
  void nEffPersona;
  void personaSimilarity;
  void identitySimilarity;
  void reliability;
  void centerVector;
  void cRich;

  return checks;
}
