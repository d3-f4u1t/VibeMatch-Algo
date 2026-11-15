import { useMemo, useState } from 'react';
import { Copy, Check, ArrowRightLeft } from 'lucide-react';
import {
  ClusterScatterChart,
  ExposureDistributionChart,
  ReliabilityAndDampingCharts,
} from './components/Charts';
import { ControlDeck } from './components/ControlDeck';
import { PYTHON_REFERENCE_MODULES } from './algo/pythonReference';
import { compatibilityScore, LAYERS } from './algo/scoring';
import {
  DEFAULT_CONFIG,
  DEFAULT_SCENARIO,
  MarketScenarioParams,
  runSimulation,
} from './algo/simulator';
import { GameTheoryConfig, LayerName } from './algo/types';

type ViewTab = 'pipeline' | 'clustering' | 'stable' | 'python';

export default function App() {
  const [activeTab, setActiveTab] = useState<ViewTab>('pipeline');
  const [scenario, setScenario] = useState<MarketScenarioParams>(DEFAULT_SCENARIO);
  const [config, setConfig] = useState<GameTheoryConfig>(DEFAULT_CONFIG);
  const [selectedUserId, setSelectedUserId] = useState<string>('u_001');
  const [selectedCandidateId, setSelectedCandidateId] = useState<string | null>(null);
  const [activeSnippetId, setActiveSnippetId] = useState<string>(PYTHON_REFERENCE_MODULES[0].id);
  const [copiedId, setCopiedId] = useState<string | null>(null);

  const snapshot = useMemo(() => runSimulation(scenario, config), [scenario, config]);

  const selectedUser = useMemo(
    () => snapshot.users.find((u) => u.id === selectedUserId) || snapshot.users[0],
    [snapshot.users, selectedUserId]
  );

  const userQueue = snapshot.queues[selectedUser.id] || [];
  const baseQueue = snapshot.baselineTopMatches[selectedUser.id] || [];

  const inspectedCandidateId = useMemo(() => {
    if (
      selectedCandidateId &&
      snapshot.users.some((u) => u.id === selectedCandidateId && u.id !== selectedUser.id)
    ) {
      return selectedCandidateId;
    }
    return userQueue[0]?.userId || baseQueue[0]?.userId || snapshot.users[1]?.id;
  }, [selectedCandidateId, snapshot.users, selectedUser.id, userQueue, baseQueue]);

  const inspectedCandidate = snapshot.users.find((u) => u.id === inspectedCandidateId);

  const pairwiseDetail = useMemo(() => {
    if (!inspectedCandidate) return null;
    const cq =
      ((snapshot.clustering.userClusterQuality[selectedUser.id] ?? 0.7) +
        (snapshot.clustering.userClusterQuality[inspectedCandidate.id] ?? 0.7)) /
      2;
    return compatibilityScore(
      selectedUser,
      inspectedCandidate,
      snapshot.behaviorStats,
      snapshot.weights,
      snapshot.populationMeans,
      cq
    );
  }, [inspectedCandidate, selectedUser, snapshot]);

  const handleRunNextBatch = () => {
    setScenario((prev) => ({ ...prev, seed: prev.seed + 1 }));
  };

  const handleReset = () => {
    setScenario(DEFAULT_SCENARIO);
    setConfig(DEFAULT_CONFIG);
    setSelectedUserId('u_001');
    setSelectedCandidateId(null);
  };

  const handleCopySnippet = async (id: string, code: string) => {
    try {
      await navigator.clipboard.writeText(code);
      setCopiedId(id);
      window.setTimeout(() => setCopiedId(null), 1600);
    } catch {
      setCopiedId(null);
    }
  };

  const totalBlocking =
    snapshot.metrics.bipartiteBlockingPairs + snapshot.metrics.generalBlockingPairs;

  return (
    <div className="min-h-screen bg-[#F8FAFC] text-[#0B0B0C]">
      <header className="bg-white border-b border-slate-200">
        <div className="max-w-[1360px] mx-auto px-6 h-16 flex items-center justify-between gap-8">
          <a
            href="#top"
            onClick={(e) => {
              e.preventDefault();
              setActiveTab('pipeline');
            }}
            className="text-lg font-bold tracking-tight text-[#0B0B0C] whitespace-nowrap shrink-0 font-display"
          >
            VibeMatch
          </a>

          <nav className="flex items-center gap-6 overflow-x-auto">
            {(
              [
                ['pipeline', 'Pipeline'],
                ['clustering', 'Clustering'],
                ['stable', 'Stable Queue'],
                ['python', 'Python Spec'],
              ] as [ViewTab, string][]
            ).map(([tab, label]) => (
              <button
                key={tab}
                type="button"
                onClick={() => setActiveTab(tab)}
                className={`text-sm font-medium whitespace-nowrap shrink-0 py-1 border-b-2 transition-colors cursor-pointer ${
                  activeTab === tab
                    ? 'border-[#FF3D5C] text-[#0B0B0C]'
                    : 'border-transparent text-slate-600 hover:text-[#0B0B0C]'
                }`}
              >
                {label}
              </button>
            ))}
          </nav>

          <button
            type="button"
            onClick={handleRunNextBatch}
            className="px-4 py-2 text-xs font-semibold text-white bg-[#FF3D5C] hover:bg-[#E63250] rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer"
          >
            Run Batch
          </button>
        </div>
      </header>

      <main className="max-w-[1360px] mx-auto px-6 py-8 space-y-8">
        <section className="space-y-6">
          <div className="max-w-3xl space-y-2">
            <p className="text-xs font-medium text-slate-500">
              Multi-Layer Shrinkage · Cold-Start-to-Attraction Sets · Batched Many-to-Many Gale-Shapley
            </p>
            <h1 className="text-3xl font-bold tracking-tight text-[#0B0B0C] text-balance">
              Upgraded VibeMatch Matching & Stable Recommendation Engine
            </h1>
            <p className="text-[15px] leading-relaxed text-slate-600">
              Live algorithmic benchmark comparing VibeMatch&apos;s legacy greedy baseline against the repaired 5-layer shrinkage scorer, DBSCAN attraction clustering, selectivity-damped exposure budgets, and capacity-K Deferred Acceptance queues.
            </p>
          </div>

          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="bg-white border border-slate-200 rounded-lg p-4">
              <div className="text-xs text-slate-500">Mutual Queue Pairs</div>
              <div className="mt-1 flex items-baseline gap-2 font-mono tabular-nums">
                <span className="text-2xl font-semibold text-[#0B0B0C]">
                  {snapshot.metrics.mutualMatchesUpgraded}
                </span>
                <span className="text-xs text-slate-500">
                  vs {snapshot.metrics.mutualMatchesBaseline} baseline
                </span>
              </div>
            </div>

            <div className="bg-white border border-slate-200 rounded-lg p-4">
              <div className="text-xs text-slate-500">Exposure Gini (Lower = Fairer)</div>
              <div className="mt-1 flex items-baseline gap-2 font-mono tabular-nums">
                <span className="text-2xl font-semibold text-[#0284C7]">
                  {snapshot.metrics.exposureGiniUpgraded.toFixed(3)}
                </span>
                <span className="text-xs text-slate-500">
                  vs {snapshot.metrics.exposureGiniBaseline.toFixed(3)} baseline
                </span>
              </div>
            </div>

            <div className="bg-white border border-slate-200 rounded-lg p-4">
              <div className="text-xs text-slate-500">Cold-Start Surface Rate</div>
              <div className="mt-1 flex items-baseline gap-2 font-mono tabular-nums">
                <span className="text-2xl font-semibold text-[#059669]">
                  {Math.round(snapshot.metrics.coldStartCoverageUpgraded * 100)}%
                </span>
                <span className="text-xs text-slate-500">
                  vs {Math.round(snapshot.metrics.coldStartCoverageBaseline * 100)}% baseline
                </span>
              </div>
            </div>

            <div className="bg-white border border-slate-200 rounded-lg p-4">
              <div className="text-xs text-slate-500">Stable Blocking Pairs</div>
              <div className="mt-1 flex items-baseline gap-2 font-mono tabular-nums">
                <span
                  className={`text-2xl font-semibold ${
                    totalBlocking === 0 ? 'text-[#059669]' : 'text-[#D97706]'
                  }`}
                >
                  {totalBlocking}
                </span>
                <span className="text-xs text-slate-500">
                  bi={snapshot.metrics.bipartiteBlockingPairs} · gen={snapshot.metrics.generalBlockingPairs}
                </span>
              </div>
            </div>
          </div>
        </section>

        <div className="grid grid-cols-1 lg:grid-cols-[minmax(0,1.68fr)_minmax(320px,1fr)] gap-8 items-start">
          <div className="space-y-6 min-w-0">
            {activeTab === 'pipeline' && (
              <>
                <section className="bg-white border border-slate-200 rounded-lg p-5 space-y-4">
                  <div className="flex flex-wrap items-baseline justify-between gap-2">
                    <div>
                      <h2 className="text-base font-semibold text-slate-900">
                        Empirical AUC Layer Weights & Population Shrinkage Priors
                      </h2>
                      <p className="text-xs text-slate-500 mt-0.5">
                        Derived from settled mutual outcomes blended with uniform prior (kappa={config.priorStrengthKappa}), never hand-tuned runtime constants.
                      </p>
                    </div>
                  </div>

                  <div className="grid grid-cols-5 gap-3 pt-1">
                    {LAYERS.map((layer) => {
                      const w = snapshot.weights[layer];
                      const mu = snapshot.populationMeans[layer];
                      return (
                        <div key={layer} className="border border-slate-200 rounded-md p-3 space-y-1.5">
                          <div className="text-xs font-medium text-slate-700 capitalize">
                            {layer}
                          </div>
                          <div className="text-lg font-mono font-semibold tabular-nums text-[#0B0B0C]">
                            {(w * 100).toFixed(1)}%
                          </div>
                          <div className="w-full h-1.5 bg-slate-100 rounded-sm overflow-hidden">
                            <div
                              className="h-full bg-[#0B0B0C]"
                              style={{ width: `${Math.min(100, w * 220)}%` }}
                            />
                          </div>
                          <div className="text-[11px] font-mono text-slate-500 tabular-nums">
                            mu={mu.toFixed(3)}
                          </div>
                        </div>
                      );
                    })}
                  </div>

                  <ReliabilityAndDampingCharts
                    spamDampingGamma={config.spamDampingGamma}
                    selectedUser={selectedUser}
                    snapshot={snapshot}
                  />
                </section>

                <section className="bg-white border border-slate-200 rounded-lg p-5 space-y-5">
                  <div className="flex flex-wrap items-center justify-between gap-3">
                    <div>
                      <h2 className="text-base font-semibold text-slate-900">
                        Pairwise Directional & Reciprocal Breakdown
                      </h2>
                      <p className="text-xs text-slate-500 mt-0.5">
                        Viewer: {selectedUser.name} ({selectedUser.city} · {selectedUser.age} · {selectedUser.archetype.replace('_', ' ')})
                      </p>
                    </div>

                    <select
                      value={inspectedCandidateId || ''}
                      onChange={(e) => setSelectedCandidateId(e.target.value)}
                      aria-label="Select candidate for pairwise inspection"
                      className="px-3 py-1.5 text-xs bg-slate-50 border border-slate-200 rounded-md text-slate-900"
                    >
                      {userQueue.map((q) => {
                        const cu = snapshot.users.find((u) => u.id === q.userId);
                        return (
                          <option key={q.userId} value={q.userId}>
                            #{q.rank} {cu?.name || q.userId} · M={q.compatibility.toFixed(3)} · {q.source}
                          </option>
                        );
                      })}
                      {snapshot.users
                        .filter((u) => u.id !== selectedUser.id && !userQueue.some((q) => q.userId === u.id))
                        .slice(0, 10)
                        .map((u) => (
                          <option key={u.id} value={u.id}>
                            Pool: {u.name} ({u.city})
                          </option>
                        ))}
                    </select>
                  </div>

                  {inspectedCandidate && pairwiseDetail && (
                    <>
                      <div className="flex flex-wrap items-center justify-between gap-4 py-3 px-4 bg-slate-50 border border-slate-200 rounded-md">
                        <div className="space-y-0.5">
                          <div className="text-sm font-semibold text-slate-900">
                            {selectedUser.name} <ArrowRightLeft className="inline w-3.5 h-3.5 mx-1 text-slate-400" /> {inspectedCandidate.name}
                          </div>
                          <div className="text-xs text-slate-600">
                            {pairwiseDetail.reason}
                          </div>
                        </div>
                        <div className="flex items-center gap-5 font-mono text-xs tabular-nums">
                          <div>
                            <span className="text-slate-500">A→B </span>
                            <span className="font-semibold text-slate-900">{pairwiseDetail.forward.toFixed(3)}</span>
                          </div>
                          <div>
                            <span className="text-slate-500">B→A </span>
                            <span className="font-semibold text-slate-900">{pairwiseDetail.reverse.toFixed(3)}</span>
                          </div>
                          <div>
                            <span className="text-slate-500">Harmonic </span>
                            <span className="font-semibold text-[#FF3D5C]">{pairwiseDetail.total.toFixed(3)}</span>
                          </div>
                        </div>
                      </div>

                      <div className="overflow-x-auto">
                        <table className="w-full text-left border-collapse text-xs">
                          <thead>
                            <tr className="border-b border-slate-200 text-slate-500">
                              <th className="py-2 pr-3 font-medium">Layer</th>
                              <th className="py-2 px-3 font-medium">Raw A→B</th>
                              <th className="py-2 px-3 font-medium">Raw B→A</th>
                              <th className="py-2 px-3 font-medium">Pair lam</th>
                              <th className="py-2 px-3 font-medium">Shrunk A→B</th>
                              <th className="py-2 px-3 font-medium">Weight</th>
                              <th className="py-2 pl-3 font-medium text-right">Contrib</th>
                            </tr>
                          </thead>
                          <tbody className="divide-y divide-slate-100 font-mono tabular-nums">
                            {LAYERS.map((layer: LayerName) => {
                              const fwd = pairwiseDetail.layersForward[layer];
                              const rev = pairwiseDetail.layersReverse[layer];
                              return (
                                <tr key={layer}>
                                  <td className="py-2 pr-3 font-sans font-medium text-slate-900 capitalize">
                                    {layer}
                                  </td>
                                  <td className="py-2 px-3 text-slate-700">{fwd.raw.toFixed(3)}</td>
                                  <td className="py-2 px-3 text-slate-700">{rev.raw.toFixed(3)}</td>
                                  <td className="py-2 px-3 text-[#0284C7]">{fwd.lambda.toFixed(3)}</td>
                                  <td className="py-2 px-3 text-slate-900">{fwd.shrunk.toFixed(3)}</td>
                                  <td className="py-2 px-3 text-slate-600">{fwd.weight.toFixed(3)}</td>
                                  <td className="py-2 pl-3 text-right font-semibold text-slate-900">
                                    {fwd.contribution.toFixed(4)}
                                  </td>
                                </tr>
                              );
                            })}
                          </tbody>
                        </table>
                      </div>
                    </>
                  )}
                </section>
              </>
            )}

            {activeTab === 'clustering' && (
              <>
                <ClusterScatterChart
                  snapshot={snapshot}
                  selectedUserId={selectedUser.id}
                  onSelectUser={setSelectedUserId}
                />

                <section className="bg-white border border-slate-200 rounded-lg p-5 space-y-4">
                  <div className="flex flex-wrap items-baseline justify-between gap-2">
                    <div>
                      <h2 className="text-base font-semibold text-slate-900">
                        Overlapping Multi-Set Registry & Quality Gates
                      </h2>
                      <p className="text-xs text-slate-500 mt-0.5">
                        Attraction clusters formed: {snapshot.clustering.attractionClustersCount} · Failed sets merged: {snapshot.clustering.prunedSetsCount}
                      </p>
                    </div>
                    <span className="text-xs text-slate-500">
                      Gates: size ≥ 10 · richness ≥ 0.30 · match_rate ≥ 0.10 · active ≤ 90d
                    </span>
                  </div>

                  <div className="overflow-x-auto">
                    <table className="w-full text-left border-collapse text-xs">
                      <thead>
                        <tr className="border-b border-slate-200 text-slate-500">
                          <th className="py-2 pr-3 font-medium">Set</th>
                          <th className="py-2 px-3 font-medium">Type</th>
                          <th className="py-2 px-3 font-medium text-right">Size</th>
                          <th className="py-2 px-3 font-medium text-right">Richness</th>
                          <th className="py-2 px-3 font-medium text-right">Match Rate</th>
                          <th className="py-2 pl-3 font-medium text-right">Gate Status</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-100">
                        {snapshot.clustering.sets.map((s) => (
                          <tr key={s.id}>
                            <td className="py-2 pr-3 font-medium text-slate-900">{s.label}</td>
                            <td className="py-2 px-3 text-slate-600 capitalize">{s.type}</td>
                            <td className="py-2 px-3 text-right font-mono tabular-nums">{s.size}</td>
                            <td className="py-2 px-3 text-right font-mono tabular-nums">
                              {s.avgRichness.toFixed(2)}
                            </td>
                            <td className="py-2 px-3 text-right font-mono tabular-nums">
                              {(s.matchSuccessRate * 100).toFixed(1)}%
                            </td>
                            <td className="py-2 pl-3 text-right font-mono text-[11px]">
                              {s.healthy ? (
                                <span className="text-[#059669] font-medium">HEALTHY</span>
                              ) : (
                                <span className="text-[#D97706] font-medium">
                                  {s.mergedIntoId ? 'MERGED' : 'PRUNED'} ({s.failureReasons.join(', ')})
                                </span>
                              )}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </section>
              </>
            )}

            {activeTab === 'stable' && (
              <>
                <ExposureDistributionChart snapshot={snapshot} />

                <section className="bg-white border border-slate-200 rounded-lg p-5 space-y-4">
                  <div className="flex flex-wrap items-baseline justify-between gap-2">
                    <div>
                      <h2 className="text-base font-semibold text-slate-900">
                        Top-{config.queueSizeK} Stable Queue vs. Legacy Greedy Baseline
                      </h2>
                      <p className="text-xs text-slate-500 mt-0.5">
                        Inspecting {selectedUser.name} ({selectedUser.gender} seeking {selectedUser.interestedIn.join('/')})
                      </p>
                    </div>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6 pt-2">
                    <div className="space-y-3">
                      <h3 className="text-xs font-semibold text-[#0284C7] uppercase tracking-wider">
                        Upgraded Stable Queue
                      </h3>
                      {userQueue.length === 0 ? (
                        <p className="text-xs text-slate-500">
                          No eligible candidates remain under current hard filters.
                        </p>
                      ) : (
                        <div className="divide-y divide-slate-100 border border-slate-200 rounded-md">
                          {userQueue.map((item) => {
                            const target = snapshot.users.find((u) => u.id === item.userId);
                            return (
                              <button
                                key={item.userId}
                                type="button"
                                onClick={() => {
                                  setSelectedCandidateId(item.userId);
                                  setActiveTab('pipeline');
                                }}
                                className="w-full text-left p-3 hover:bg-slate-50 transition-colors space-y-1 cursor-pointer"
                              >
                                <div className="flex items-center justify-between gap-2">
                                  <span className="text-sm font-medium text-slate-900">
                                    #{item.rank} {target?.name || item.userId}
                                  </span>
                                  <span className="text-xs font-mono tabular-nums text-slate-700">
                                    M={item.compatibility.toFixed(3)} · P={Math.round(item.expectedAcceptance * 100)}%
                                  </span>
                                </div>
                                <div className="text-xs text-slate-600">
                                  {item.reason}
                                </div>
                                <div className="text-[11px] font-mono text-slate-500">
                                  {item.source} · {target?.archetype.replace('_', ' ')} · Elo {(snapshot.eloByUserSet[item.userId] ?? 1500).toFixed(0)}
                                </div>
                              </button>
                            );
                          })}
                        </div>
                      )}
                    </div>

                    <div className="space-y-3">
                      <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wider">
                        Legacy Greedy Baseline
                      </h3>
                      {baseQueue.length === 0 ? (
                        <p className="text-xs text-slate-500">No candidates.</p>
                      ) : (
                        <div className="divide-y divide-slate-100 border border-slate-200 rounded-md">
                          {baseQueue.map((item) => {
                            const target = snapshot.users.find((u) => u.id === item.userId);
                            return (
                              <div key={item.userId} className="p-3 space-y-1">
                                <div className="flex items-center justify-between gap-2">
                                  <span className="text-sm font-medium text-slate-700">
                                    #{item.rank} {target?.name || item.userId}
                                  </span>
                                  <span className="text-xs font-mono tabular-nums text-slate-500">
                                    v1={item.compatibility.toFixed(3)}
                                  </span>
                                </div>
                                <div className="text-xs text-slate-500">
                                  Unilateral greedy max · ignores congestion &amp; selectivity
                                </div>
                                <div className="text-[11px] font-mono text-slate-400">
                                  Inbound load: {snapshot.exposureBaseline[item.userId] || 0} impressions
                                </div>
                              </div>
                            );
                          })}
                        </div>
                      )}
                    </div>
                  </div>
                </section>
              </>
            )}

            {activeTab === 'python' && (
              <section className="bg-white border border-slate-200 rounded-lg p-5 space-y-5">
                <div className="flex flex-wrap items-baseline justify-between gap-2">
                  <div>
                    <h2 className="text-base font-semibold text-slate-900">
                      FastAPI Python Reference Implementation
                    </h2>
                    <p className="text-xs text-slate-500 mt-0.5">
                      Direct drop-in algorithm modules matching the mathematical engine running live in this workbench.
                    </p>
                  </div>
                </div>

                <div className="flex flex-wrap gap-2">
                  {PYTHON_REFERENCE_MODULES.map((mod) => (
                    <button
                      key={mod.id}
                      type="button"
                      onClick={() => setActiveSnippetId(mod.id)}
                      className={`px-3 py-1.5 text-xs font-medium rounded-md border transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                        activeSnippetId === mod.id
                          ? 'bg-[#0B0B0C] text-white border-[#0B0B0C]'
                          : 'bg-slate-50 text-slate-700 border-slate-200 hover:bg-slate-100'
                      }`}
                    >
                      {mod.targetModule}
                    </button>
                  ))}
                </div>

                {(() => {
                  const mod =
                    PYTHON_REFERENCE_MODULES.find((m) => m.id === activeSnippetId) ||
                    PYTHON_REFERENCE_MODULES[0];
                  return (
                    <div className="space-y-3">
                      <div className="flex items-start justify-between gap-4">
                        <div className="space-y-1">
                          <h3 className="text-sm font-semibold text-slate-900">{mod.title}</h3>
                          <p className="text-xs text-slate-600 leading-relaxed">{mod.summary}</p>
                        </div>
                        <button
                          type="button"
                          onClick={() => handleCopySnippet(mod.id, mod.code)}
                          className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium border border-slate-200 rounded-md hover:bg-slate-50 whitespace-nowrap shrink-0 cursor-pointer"
                        >
                          {copiedId === mod.id ? (
                            <>
                              <Check className="w-3.5 h-3.5 text-[#059669]" />
                              Copied
                            </>
                          ) : (
                            <>
                              <Copy className="w-3.5 h-3.5" />
                              Copy Code
                            </>
                          )}
                        </button>
                      </div>

                      <pre className="p-4 bg-[#0B0B0C] text-slate-100 rounded-md text-xs leading-relaxed overflow-x-auto">
                        <code>{mod.code}</code>
                      </pre>
                    </div>
                  );
                })()}
              </section>
            )}
          </div>

          <ControlDeck
            scenario={scenario}
            config={config}
            snapshot={snapshot}
            selectedUserId={selectedUser.id}
            onChangeScenario={setScenario}
            onChangeConfig={setConfig}
            onSelectUser={setSelectedUserId}
            onReset={handleReset}
          />
        </div>
      </main>
    </div>
  );
}
