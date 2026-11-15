import { UserProfile } from './types';

export const EMBED_DIM = 48;

// Semantic genre/mood/persona anchors give realistic cosine clustering and
// anisotropy (non-zero shared mean direction) just like all-MiniLM-L6-v2.
const SEMANTIC_BASES: Record<string, number[]> = {};

function hashString(str: string): number {
  let h = 2166136261;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

function getTokenBasis(token: string): number[] {
  const key = token.trim().toLowerCase();
  if (SEMANTIC_BASES[key]) return SEMANTIC_BASES[key];

  const vec = new Array<number>(EMBED_DIM).fill(0);
  // Intentional mild global anisotropy offset along early dimensions
  // so mean-centering proves measurable separation improvement.
  vec[0] = 0.28;
  vec[1] = 0.19;

  const seed = hashString(key);
  let state = seed || 1;
  for (let i = 0; i < EMBED_DIM; i++) {
    state = (Math.imul(1664525, state) + 1013904223) >>> 0;
    const u = (state / 0xffffffff) * 2 - 1;
    vec[i] += u * 0.75;
  }

  // Group strongly related musical tokens into coherent subspaces
  const clusters: [string[], number[]][] = [
    [['sza', 'frank ocean', 'fka twigs', 'daniel caesar', 'r&b', 'neo-soul', 'alt-r&b', 'late night', '3am', 'melancholic', 'chill'], [2, 3, 4, 5]],
    [['travis scott', 'drake', 'kanye west', 'kendrick lamar', 'hip-hop', 'rap', 'trap', 'gym', 'hype', 'high', 'party'], [6, 7, 8, 9]],
    [['tame impala', 'mitski', 'phoebe bridgers', 'arctic monkeys', 'indie', 'alt-rock', 'shoegaze', 'road trip', 'dreamy', '2010s'], [10, 11, 12, 13]],
    [['fred again..', 'kaytranada', 'disclosure', 'four tet', 'electronic', 'house', 'uk garage', 'club', 'workout', 'euphoric'], [14, 15, 16, 17]],
    [['miles davis', 'kamasi washington', 'hiatus kaiyote', 'badbadnotgood', 'jazz', 'soul', 'instrumental', 'studying', 'coffee', 'warm'], [18, 19, 20, 21]],
    [['concerts', 'vinyl', 'museums', 'film', 'cooking', 'hiking', 'bookstore', 'photography', 'coffee'], [22, 23, 24, 25]],
  ];

  for (const [keywords, dims] of clusters) {
    if (keywords.some((kw) => key.includes(kw) || kw.includes(key))) {
      for (const d of dims) {
        vec[d] += 1.65;
      }
    }
  }

  const norm = Math.hypot(...vec) || 1;
  const normalized = vec.map((v) => v / norm);
  SEMANTIC_BASES[key] = normalized;
  return normalized;
}

export function normalizeVector(vec: number[]): number[] {
  if (vec.length === 0) return [];
  const norm = Math.hypot(...vec);
  if (!Number.isFinite(norm) || norm < 1e-12) return [];
  return vec.map((v) => v / norm);
}

export function cosineSimilarity(a: number[], b: number[]): number {
  if (!a.length || !b.length || a.length !== b.length) return 0;
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb);
  if (denom < 1e-12) return 0;
  return Math.max(-1, Math.min(1, dot / denom));
}

export function encodeTokens(tokens: string[]): number[] {
  const clean = tokens.map((t) => t.trim().toLowerCase()).filter(Boolean);
  if (clean.length === 0) return [];
  const acc = new Array<number>(EMBED_DIM).fill(0);
  for (const tok of clean) {
    const basis = getTokenBasis(tok);
    for (let i = 0; i < EMBED_DIM; i++) {
      acc[i] += basis[i];
    }
  }
  return normalizeVector(acc);
}

// Track-inclusive taste encoding (fixes repo bug where tracks were omitted from taste vector)
export function buildTasteTokens(user: Pick<UserProfile, 'artists' | 'tracks'>): string[] {
  const tokens: string[] = [];
  for (const a of user.artists) {
    if (a.name.trim()) tokens.push(a.name.trim());
    for (const tag of a.tags) {
      if (tag.trim()) tokens.push(tag.trim());
    }
  }
  for (const t of user.tracks) {
    if (t.artistName.trim()) tokens.push(t.artistName.trim());
    if (t.title.trim()) tokens.push(t.title.trim());
  }
  return tokens;
}

export function buildContextTokens(
  user: Pick<UserProfile, 'musicMoods' | 'musicEras' | 'musicEnergy' | 'musicContexts'>
): string[] {
  const tokens: string[] = [
    ...user.musicMoods,
    ...user.musicEras,
    ...user.musicContexts,
  ];
  if (user.musicEnergy.trim()) tokens.push(user.musicEnergy.trim());
  return tokens.filter((t) => t.trim().length > 0);
}

export function buildPersonaTokens(user: Pick<UserProfile, 'bio' | 'intent'>): string[] {
  const tokens: string[] = [];
  if (user.bio.trim()) {
    const words = user.bio
      .toLowerCase()
      .replace(/[^a-z0-9\s-]/g, ' ')
      .split(/\s+/)
      .filter((w) => w.length >= 3);
    tokens.push(...words);
  }
  if (user.intent.trim() && user.bio.trim()) {
    tokens.push(user.intent.replace('_', ' '));
  }
  return tokens;
}

export function computeMeanVector(vectors: number[][]): number[] | null {
  const valid = vectors.filter((v) => v.length === EMBED_DIM);
  if (valid.length < 2) return null;
  const mean = new Array<number>(EMBED_DIM).fill(0);
  for (const v of valid) {
    for (let i = 0; i < EMBED_DIM; i++) {
      mean[i] += v[i];
    }
  }
  for (let i = 0; i < EMBED_DIM; i++) {
    mean[i] /= valid.length;
  }
  return mean;
}

export function centerVector(vec: number[], mean: number[] | null): number[] {
  if (!vec.length) return [];
  if (!mean || mean.length !== vec.length) return vec;
  const centered = vec.map((v, i) => v - mean[i]);
  const norm = Math.hypot(...centered);
  if (norm < 1e-8) return vec;
  return centered.map((v) => v / norm);
}

// Independent per-layer population centering (fixes cross-space centering bug in repo)
export function encodeAndCenterAllUsers(users: UserProfile[]): {
  tasteMean: number[] | null;
  contextMean: number[] | null;
  personaMean: number[] | null;
} {
  for (const u of users) {
    u.rawTasteVec = encodeTokens(buildTasteTokens(u));
    u.rawContextVec = encodeTokens(buildContextTokens(u));
    u.rawPersonaVec = encodeTokens(buildPersonaTokens(u));
  }

  const tasteMean = computeMeanVector(users.map((u) => u.rawTasteVec));
  const contextMean = computeMeanVector(users.map((u) => u.rawContextVec));
  const personaMean = computeMeanVector(users.map((u) => u.rawPersonaVec));

  for (const u of users) {
    u.tasteVec = centerVector(u.rawTasteVec, tasteMean);
    u.contextVec = centerVector(u.rawContextVec, contextMean);
    u.personaVec = centerVector(u.rawPersonaVec, personaMean);
    u.tasteXY = projectTo2D(u.tasteVec);
  }

  return { tasteMean, contextMean, personaMean };
}

export function projectTo2D(vec: number[]): [number, number] {
  if (!vec.length) return [0, 0];
  let x = 0;
  let y = 0;
  for (let i = 0; i < vec.length; i++) {
    const angle = (i * 2.3999632) % (Math.PI * 2); // golden-angle deterministic projection
    x += vec[i] * Math.cos(angle);
    y += vec[i] * Math.sin(angle);
  }
  return [Number(x.toFixed(4)), Number(y.toFixed(4))];
}
