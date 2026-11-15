import { LAYER_HALF_LIFE, reliability, selectivityDamping } from '../algo/scoring';
import { LayerName, SimulationSnapshot, UserProfile } from '../algo/types';

export function ReliabilityAndDampingCharts({
  spamDampingGamma,
  selectedUser,
  snapshot,
}: {
  spamDampingGamma: number;
  selectedUser: UserProfile;
  snapshot: SimulationSnapshot;
}) {
  // SVG 1: Saturating reliability curves lambda = n / (n + k)
  const w = 360;
  const h = 160;
  const padL = 34;
  const padR = 12;
  const padT = 14;
  const padB = 26;
  const plotW = w - padL - padR;
  const plotH = h - padT - padB;

  const layers: { key: LayerName; label: string; color: string }[] = [
    { key: 'persona', label: 'Persona (k=1)', color: '#059669' },
    { key: 'context', label: 'Context (k=4)', color: '#0284C7' },
    { key: 'taste', label: 'Taste (k=6)', color: '#0B0B0C' },
    { key: 'behavior', label: 'Behavior (k=25)', color: '#FF3D5C' },
  ];

  const makeRelPath = (k: number) => {
    const pts: string[] = [];
    for (let n = 0; n <= 40; n += 1) {
      const x = padL + (n / 40) * plotW;
      const y = padT + (1 - reliability(n, k)) * plotH;
      pts.push(`${n === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`);
    }
    return pts.join(' ');
  };

  // SVG 2: Selectivity damping phi(r_u)
  const makeDampPath = () => {
    const pts: string[] = [];
    for (let i = 0; i <= 50; i += 1) {
      const r = i / 50;
      const phi = selectivityDamping(r, 30, spamDampingGamma);
      const x = padL + r * plotW;
      const y = padT + (1 - phi) * plotH;
      pts.push(`${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`);
    }
    return pts.join(' ');
  };

  const uStats = snapshot.behaviorStats[selectedUser.id];
  const userRate = uStats?.rightSwipeRate ?? 0.35;
  const userPhi = uStats?.selectivityDamping ?? 1.0;
  const dotX = padL + userRate * plotW;
  const dotY = padT + (1 - userPhi) * plotH;

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 pt-4">
      <div className="bg-white border border-slate-200 rounded-lg p-4">
        <div className="flex items-baseline justify-between mb-2">
          <h3 className="text-sm font-semibold text-slate-900">
            Shrinkage Reliability Curve
          </h3>
          <span className="text-xs font-mono text-slate-500">
            lam = n_eff / (n_eff + k)
          </span>
        </div>
        <svg viewBox={`0 0 ${w} ${h}`} className="w-full h-40">
          {[0, 0.5, 1].map((tick) => {
            const y = padT + (1 - tick) * plotH;
            return (
              <g key={tick}>
                <line
                  x1={padL}
                  y1={y}
                  x2={w - padR}
                  y2={y}
                  stroke="#E2E8F0"
                  strokeDasharray={tick === 0 ? undefined : '3,3'}
                />
                <text
                  x={padL - 6}
                  y={y + 3}
                  textAnchor="end"
                  className="fill-slate-500 text-[9px] font-mono"
                >
                  {tick.toFixed(1)}
                </text>
              </g>
            );
          })}
          {[0, 10, 20, 30, 40].map((n) => {
            const x = padL + (n / 40) * plotW;
            return (
              <text
                key={n}
                x={x}
                y={h - 6}
                textAnchor="middle"
                className="fill-slate-500 text-[9px] font-mono"
              >
                {n}
              </text>
            );
          })}
          {layers.map((l) => (
            <path
              key={l.key}
              d={makeRelPath(LAYER_HALF_LIFE[l.key])}
              fill="none"
              stroke={l.color}
              strokeWidth={1.75}
            />
          ))}
        </svg>
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1 mt-2 text-xs text-slate-600">
          {layers.map((l) => (
            <span key={l.key} className="inline-flex items-center gap-1.5">
              <span
                className="inline-block w-3 h-0.5"
                style={{ backgroundColor: l.color }}
              />
              {l.label}
            </span>
          ))}
        </div>
      </div>

      <div className="bg-white border border-slate-200 rounded-lg p-4">
        <div className="flex items-baseline justify-between mb-2">
          <h3 className="text-sm font-semibold text-slate-900">
            Selectivity Damping Function
          </h3>
          <span className="text-xs font-mono text-slate-500">
            phi(r) · active={selectedUser.name}
          </span>
        </div>
        <svg viewBox={`0 0 ${w} ${h}`} className="w-full h-40">
          <rect
            x={padL + 0.15 * plotW}
            y={padT}
            width={0.4 * plotW}
            height={plotH}
            fill="#0284C7"
            fillOpacity={0.07}
          />
          {[0, 0.5, 1].map((tick) => {
            const y = padT + (1 - tick) * plotH;
            return (
              <g key={tick}>
                <line
                  x1={padL}
                  y1={y}
                  x2={w - padR}
                  y2={y}
                  stroke="#E2E8F0"
                  strokeDasharray={tick === 0 ? undefined : '3,3'}
                />
                <text
                  x={padL - 6}
                  y={y + 3}
                  textAnchor="end"
                  className="fill-slate-500 text-[9px] font-mono"
                >
                  {tick.toFixed(1)}
                </text>
              </g>
            );
          })}
          {[0, 0.25, 0.55, 0.75, 1].map((r) => {
            const x = padL + r * plotW;
            return (
              <text
                key={r}
                x={x}
                y={h - 6}
                textAnchor="middle"
                className="fill-slate-500 text-[9px] font-mono"
              >
                {Math.round(r * 100)}%
              </text>
            );
          })}
          <path d={makeDampPath()} fill="none" stroke="#FF3D5C" strokeWidth={2} />
          <circle cx={dotX} cy={dotY} r={4.5} fill="#0B0B0C" stroke="#FFFFFF" strokeWidth={1.5} />
        </svg>
        <div className="flex items-center justify-between mt-2 text-xs text-slate-600">
          <span>Healthy zone: 15%–55% right-swipe rate</span>
          <span className="font-mono tabular-nums text-slate-900">
            r={Math.round(userRate * 100)}% / phi={userPhi.toFixed(2)}
          </span>
        </div>
      </div>
    </div>
  );
}

const ARCHETYPE_COLORS: Record<string, string> = {
  selective: '#059669',
  balanced: '#0284C7',
  cold_start: '#D97706',
  spammer: '#FF3D5C',
};

export function ClusterScatterChart({
  snapshot,
  selectedUserId,
  onSelectUser,
}: {
  snapshot: SimulationSnapshot;
  selectedUserId: string;
  onSelectUser: (id: string) => void;
}) {
  const w = 640;
  const h = 340;
  const pad = 32;

  const xs = snapshot.users.map((u) => u.tasteXY[0]);
  const ys = snapshot.users.map((u) => u.tasteXY[1]);
  const minX = Math.min(...xs, -1);
  const maxX = Math.max(...xs, 1);
  const minY = Math.min(...ys, -1);
  const maxY = Math.max(...ys, 1);
  const spanX = maxX - minX || 1;
  const spanY = maxY - minY || 1;

  const mapX = (v: number) => pad + ((v - minX) / spanX) * (w - 2 * pad);
  const mapY = (v: number) => h - pad - ((v - minY) / spanY) * (h - 2 * pad);

  const selectedQueueIds = new Set(
    (snapshot.queues[selectedUserId] || []).map((q) => q.userId)
  );
  const selectedUser = snapshot.users.find((u) => u.id === selectedUserId) || snapshot.users[0];

  return (
    <div className="bg-white border border-slate-200 rounded-lg p-5">
      <div className="flex flex-wrap items-baseline justify-between gap-2 mb-3">
        <div>
          <h3 className="text-base font-semibold text-slate-900">
            Taste & Attraction Manifold Projection
          </h3>
          <p className="text-xs text-slate-500 mt-0.5">
            Click any user node to inspect multi-set memberships and stable queue edges.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-4 text-xs text-slate-600">
          <span>Selective (emerald)</span>
          <span>·</span>
          <span>Balanced (azure)</span>
          <span>·</span>
          <span>Cold-start (amber)</span>
          <span>·</span>
          <span>Spammer (pop)</span>
        </div>
      </div>

      <svg viewBox={`0 0 ${w} ${h}`} className="w-full h-80 bg-slate-50/60 border border-slate-100 rounded">
        <line x1={pad} y1={h / 2} x2={w - pad} y2={h / 2} stroke="#E2E8F0" strokeDasharray="3,3" />
        <line x1={w / 2} y1={pad} x2={w / 2} y2={h - pad} stroke="#E2E8F0" strokeDasharray="3,3" />

        {selectedUser &&
          snapshot.users
            .filter((u) => selectedQueueIds.has(u.id))
            .map((target) => (
              <line
                key={target.id}
                x1={mapX(selectedUser.tasteXY[0])}
                y1={mapY(selectedUser.tasteXY[1])}
                x2={mapX(target.tasteXY[0])}
                y2={mapY(target.tasteXY[1])}
                stroke="#FF3D5C"
                strokeOpacity={0.45}
                strokeWidth={1.5}
              />
            ))}

        {snapshot.users.map((u) => {
          const cx = mapX(u.tasteXY[0]);
          const cy = mapY(u.tasteXY[1]);
          const isSelected = u.id === selectedUserId;
          const isQueued = selectedQueueIds.has(u.id);
          return (
            <g
              key={u.id}
              onClick={() => onSelectUser(u.id)}
              className="cursor-pointer"
            >
              <circle
                cx={cx}
                cy={cy}
                r={isSelected ? 7.5 : isQueued ? 5.5 : 4}
                fill={ARCHETYPE_COLORS[u.archetype] || '#0284C7'}
                fillOpacity={isSelected || isQueued ? 1 : 0.78}
                stroke={isSelected ? '#0B0B0C' : isQueued ? '#FF3D5C' : '#FFFFFF'}
                strokeWidth={isSelected ? 2.2 : isQueued ? 1.5 : 0.8}
              />
            </g>
          );
        })}
      </svg>
    </div>
  );
}

export function ExposureDistributionChart({
  snapshot,
}: {
  snapshot: SimulationSnapshot;
}) {
  const w = 640;
  const h = 210;
  const padL = 42;
  const padR = 16;
  const padT = 16;
  const padB = 28;
  const plotW = w - padL - padR;
  const plotH = h - padT - padB;

  const baseSorted = Object.values(snapshot.exposureBaseline).sort((a, b) => b - a);
  const upgSorted = Object.values(snapshot.exposureAssigned).sort((a, b) => b - a);
  const maxVal = Math.max(...baseSorted, ...upgSorted, 1);

  const buildPath = (arr: number[]) =>
    arr
      .map((v, i) => {
        const x = padL + (i / Math.max(1, arr.length - 1)) * plotW;
        const y = padT + (1 - v / maxVal) * plotH;
        return `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(' ');

  return (
    <div className="bg-white border border-slate-200 rounded-lg p-5">
      <div className="flex flex-wrap items-baseline justify-between gap-2 mb-3">
        <div>
          <h3 className="text-base font-semibold text-slate-900">
            Inbound Impression Concentration (Rank-Ordered Users)
          </h3>
          <p className="text-xs text-slate-500 mt-0.5">
            Exposure budgets flatten top-10% winner-take-all congestion compared to greedy ranking.
          </p>
        </div>
        <div className="flex items-center gap-4 text-xs font-mono tabular-nums">
          <span className="text-slate-500">
            Baseline Gini: {snapshot.metrics.exposureGiniBaseline.toFixed(3)}
          </span>
          <span className="text-[#0284C7] font-medium">
            Upgraded Gini: {snapshot.metrics.exposureGiniUpgraded.toFixed(3)}
          </span>
        </div>
      </div>

      <svg viewBox={`0 0 ${w} ${h}`} className="w-full h-52">
        {[0, 0.5, 1].map((frac) => {
          const val = Math.round(frac * maxVal);
          const y = padT + (1 - frac) * plotH;
          return (
            <g key={frac}>
              <line x1={padL} y1={y} x2={w - padR} y2={y} stroke="#E2E8F0" strokeDasharray="3,3" />
              <text x={padL - 8} y={y + 3} textAnchor="end" className="fill-slate-500 text-[10px] font-mono">
                {val}
              </text>
            </g>
          );
        })}
        <path d={buildPath(baseSorted)} fill="none" stroke="#94A3B8" strokeWidth={1.75} strokeDasharray="4,3" />
        <path d={buildPath(upgSorted)} fill="none" stroke="#0284C7" strokeWidth={2.25} />
        <text x={padL} y={h - 6} className="fill-slate-500 text-[10px]">
          Most exposed profiles
        </text>
        <text x={w - padR} y={h - 6} textAnchor="end" className="fill-slate-500 text-[10px]">
          Long-tail profiles
        </text>
      </svg>

      <div className="flex items-center gap-6 mt-2 text-xs text-slate-600">
        <span className="inline-flex items-center gap-2">
          <span className="inline-block w-4 h-0.5 border-t-2 border-dashed border-slate-400" />
          Legacy Greedy Baseline
        </span>
        <span className="inline-flex items-center gap-2">
          <span className="inline-block w-4 h-0.5 bg-[#0284C7]" />
          Exposure-Budgeted Gale-Shapley
        </span>
      </div>
    </div>
  );
}
