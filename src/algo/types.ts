export type LayerName = 'taste' | 'context' | 'persona' | 'identity' | 'behavior';

export type Gender = 'man' | 'woman' | 'nonbinary';
export type Intent = 'long_term' | 'friendship' | 'short_term' | 'open';
export type SwipeAction = 'LIKE' | 'PASS' | 'SUPER_LIKE';
export type Archetype = 'selective' | 'balanced' | 'spammer' | 'cold_start';

export interface ArtistPick {
  name: string;
  tags: string[];
}

export interface TrackPick {
  title: string;
  artistName: string;
}

export interface UserProfile {
  id: string;
  name: string;
  age: number;
  gender: Gender;
  interestedIn: Gender[];
  city: string;
  intent: Intent;
  ageMin: number;
  ageMax: number;
  maxDistanceKm: number;
  dealbreakers: string[];
  habits: Record<string, string>;
  artists: ArtistPick[];
  tracks: TrackPick[];
  musicMoods: string[];
  musicEras: string[];
  musicEnergy: string;
  musicContexts: string[];
  bio: string;
  archetype: Archetype;
  activeConversations: number;
  daysSinceActive: number;
  // Derived embeddings (raw + centered per layer)
  rawTasteVec: number[];
  rawContextVec: number[];
  rawPersonaVec: number[];
  tasteVec: number[];
  contextVec: number[];
  personaVec: number[];
  // 2D projection coordinates for cluster visualization
  tasteXY: [number, number];
}

export interface SwipeEvent {
  swiperId: string;
  targetId: string;
  action: SwipeAction;
  ageDays: number;
}

export interface ConversationRecord {
  userA: string;
  userB: string;
  speakers: string[];
}

export interface UserBehaviorStats {
  userId: string;
  nEff: number;
  lambdaBehavior: number;
  totalSwipes: number;
  likeCount: number;
  passCount: number;
  superLikeCount: number;
  rightSwipeRate: number;
  selectivityDamping: number;
  likedCentroid: number[] | null;
  passedCentroid: number[] | null;
  likedSpread: number;
  attractionSignature: number[] | null;
}

export interface LayerDetail {
  raw: number;
  lambda: number;
  shrunk: number;
  mu: number;
  weight: number;
  contribution: number;
}

export interface MatchDriver {
  factor: LayerName;
  label: string;
  weight: number;
  rawScore: number;
  reliability: number;
  impact: number;
}

export interface CompatibilityResult {
  total: number;
  forward: number;
  reverse: number;
  layersForward: Record<LayerName, LayerDetail>;
  layersReverse: Record<LayerName, LayerDetail>;
  weights: Record<LayerName, number>;
  drivers: MatchDriver[];
  reason: string;
  clusterQuality: number;
}

export type SetType = 'demographic' | 'taste' | 'behavior' | 'attraction';

export interface UserSet {
  id: string;
  type: SetType;
  label: string;
  memberIds: string[];
  memberWeights: Record<string, number>;
  centroid: number[] | null;
  size: number;
  avgRichness: number;
  matchSuccessRate: number;
  daysSinceActive: number;
  healthy: boolean;
  failureReasons: string[];
  mergedIntoId: string | null;
}

export interface ClusteringResult {
  sets: UserSet[];
  userSets: Record<string, { setId: string; type: SetType; weight: number; label: string }[]>;
  userClusterQuality: Record<string, number>;
  attractionClustersCount: number;
  prunedSetsCount: number;
}

export interface QueueCandidate {
  userId: string;
  compatibility: number;
  expectedAcceptance: number;
  utility: number;
  rank: number;
  source: 'stable_bipartite' | 'stable_general' | 'fallback';
  reason: string;
  drivers: MatchDriver[];
  sharedArtists: string[];
  sharedTracks: string[];
}

export interface GameTheoryConfig {
  queueSizeK: number;
  candidateTopM: number;
  exposureBudgetBase: number;
  spamDampingGamma: number;
  eloSensitivity: number;
  congestionPenalty: number;
  diversityPenalty: number;
  settlingDays: number;
  priorStrengthKappa: number;
  dbscanEps: number;
  dbscanMinSamples: number;
}

export interface MarketMetrics {
  userCount: number;
  mutualMatchesUpgraded: number;
  mutualMatchesBaseline: number;
  meanMutualCompatibility: number;
  baselineMutualCompatibility: number;
  exposureGiniUpgraded: number;
  exposureGiniBaseline: number;
  spammerMatchRateUpgraded: number;
  selectiveMatchRateUpgraded: number;
  coldStartCoverageUpgraded: number;
  coldStartCoverageBaseline: number;
  bipartiteBlockingPairs: number;
  generalBlockingPairs: number;
  runtimeMs: number;
}

export interface InvariantCheck {
  id: string;
  name: string;
  passed: boolean;
  detail: string;
  metricValue: string;
}

export interface SimulationSnapshot {
  config: GameTheoryConfig;
  users: UserProfile[];
  swipes: SwipeEvent[];
  conversations: ConversationRecord[];
  behaviorStats: Record<string, UserBehaviorStats>;
  weights: Record<LayerName, number>;
  populationMeans: Record<LayerName, number>;
  clustering: ClusteringResult;
  eloByUserSet: Record<string, number>;
  exposureAssigned: Record<string, number>;
  exposureBaseline: Record<string, number>;
  queues: Record<string, QueueCandidate[]>;
  baselineTopMatches: Record<string, QueueCandidate[]>;
  metrics: MarketMetrics;
  invariants: InvariantCheck[];
}
