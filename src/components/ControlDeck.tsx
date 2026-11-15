import { CheckCircle2, AlertTriangle, RotateCcw } from 'lucide-react';
import { MarketScenarioParams } from '../algo/simulator';
import { GameTheoryConfig, SimulationSnapshot } from '../algo/types';

interface ControlDeckProps {
  scenario: MarketScenarioParams;
  config: GameTheoryConfig;
  snapshot: SimulationSnapshot;
  selectedUserId: string;
  onChangeScenario: (next: MarketScenarioParams) => void;
  onChangeConfig: (next: GameTheoryConfig) => void;
  onSelectUser: (id: string) => void;
  onReset: () => void;
}

function SliderRow({
  label,
  valueDisplay,
  min,
  max,
  step,
  value,
  onChange,
}: {
  label: string;
  valueDisplay: string;
  min: number;
  max: number;
  step: number;
  value: number;
  onChange: (val: number) => void;
}) {
  return (
    <label className="block space-y-1">
      <div className="flex items-center justify-between text-xs">
        <span className="font-medium text-slate-700">{label}</span>
        <span className="font-mono tabular-nums text-slate-900">{valueDisplay}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full accent-[#FF3D5C] cursor-pointer"
      />
    </label>
  );
}

export function ControlDeck({
  scenario,
  config,
  snapshot,
  selectedUserId,
  onChangeScenario,
  onChangeConfig,
  onSelectUser,
  onReset,
}: ControlDeckProps) {
  const passCount = snapshot.invariants.filter((i) => i.passed).length;

  return (
    <aside className="space-y-6">
      <section className="bg-white border border-slate-200 rounded-lg p-5 space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold text-slate-900">
            Active User Inspector
          </h2>
          <span className="text-xs font-mono text-slate-500">
            {snapshot.metrics.runtimeMs} ms
          </span>
        </div>

        <select
          value={selectedUserId}
          onChange={(e) => onSelectUser(e.target.value)}
          aria-label="Select user to inspect"
          className="w-full px-3 py-2 text-sm bg-slate-50 border border-slate-200 rounded-md text-slate-900 focus:outline-none focus:ring-2 focus:ring-[#FF3D5C]/30"
        >
          {snapshot.users.slice(0, 40).map((u) => (
            <option key={u.id} value={u.id}>
              {u.name} · {u.archetype.replace('_', ' ')} · {u.gender} ({u.city})
            </option>
          ))}
        </select>

        {(() => {
          const u = snapshot.users.find((x) => x.id === selectedUserId) || snapshot.users[0];
          const st = snapshot.behaviorStats[u.id];
          const elo = snapshot.eloByUserSet[u.id] ?? 1500;
          const assigned = snapshot.exposureAssigned[u.id] ?? 0;
          return (
            <div className="grid grid-cols-2 gap-x-4 gap-y-2 pt-2 border-t border-slate-100 text-xs">
              <div>
                <div className="text-slate-500">Behavior n_eff / lam</div>
                <div className="font-mono tabular-nums text-slate-900 font-medium">
                  {st.nEff.toFixed(1)} / {st.lambdaBehavior.toFixed(2)}
                </div>
              </div>
              <div>
                <div className="text-slate-500">Selectivity phi(r)</div>
                <div className="font-mono tabular-nums text-slate-900 font-medium">
                  {Math.round(st.rightSwipeRate * 100)}% / {st.selectivityDamping.toFixed(2)}
                </div>
              </div>
              <div>
                <div className="text-slate-500">Per-Set Elo</div>
                <div className="font-mono tabular-nums text-slate-900 font-medium">
                  {elo.toFixed(1)}
                </div>
              </div>
              <div>
                <div className="text-slate-500">Inbound Exposure</div>
                <div className="font-mono tabular-nums text-slate-900 font-medium">
                  {assigned} slots
                </div>
              </div>
            </div>
          );
        })()}
      </section>

      <section className="bg-white border border-slate-200 rounded-lg p-5 space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold text-slate-900">
            Market & Mechanism Controls
          </h2>
          <button
            type="button"
            onClick={onReset}
            className="inline-flex items-center gap-1.5 text-xs font-medium text-slate-600 hover:text-slate-900 cursor-pointer whitespace-nowrap shrink-0"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            Reset
          </button>
        </div>

        <div className="space-y-3">
          <SliderRow
            label="Market Pool Size"
            valueDisplay={`${scenario.userCount} users`}
            min={60}
            max={200}
            step={20}
            value={scenario.userCount}
            onChange={(v) => onChangeScenario({ ...scenario, userCount: v })}
          />
          <SliderRow
            label="Stable Queue Capacity (K)"
            valueDisplay={`${config.queueSizeK} cards`}
            min={3}
            max={12}
            step={1}
            value={config.queueSizeK}
            onChange={(v) => onChangeConfig({ ...config, queueSizeK: v })}
          />
          <SliderRow
            label="Base Exposure Budget (B)"
            valueDisplay={`${config.exposureBudgetBase} slots`}
            min={4}
            max={18}
            step={1}
            value={config.exposureBudgetBase}
            onChange={(v) => onChangeConfig({ ...config, exposureBudgetBase: v })}
          />
          <SliderRow
            label="Selectivity Damping (gamma)"
            valueDisplay={config.spamDampingGamma.toFixed(1)}
            min={2}
            max={24}
            step={1}
            value={config.spamDampingGamma}
            onChange={(v) => onChangeConfig({ ...config, spamDampingGamma: v })}
          />
          <SliderRow
            label="Elo Gap Sensitivity"
            valueDisplay={config.eloSensitivity.toFixed(2)}
            min={0}
            max={2}
            step={0.1}
            value={config.eloSensitivity}
            onChange={(v) => onChangeConfig({ ...config, eloSensitivity: v })}
          />
          <SliderRow
            label="Swipe Spammer Share"
            valueDisplay={`${Math.round(scenario.spammerRatio * 100)}%`}
            min={0}
            max={0.4}
            step={0.05}
            value={scenario.spammerRatio}
            onChange={(v) => onChangeScenario({ ...scenario, spammerRatio: v })}
          />
          <SliderRow
            label="Cold-Start Share"
            valueDisplay={`${Math.round(scenario.coldStartRatio * 100)}%`}
            min={0.05}
            max={0.5}
            step={0.05}
            value={scenario.coldStartRatio}
            onChange={(v) => onChangeScenario({ ...scenario, coldStartRatio: v })}
          />
        </div>
      </section>

      <section className="bg-white border border-slate-200 rounded-lg p-5 space-y-3">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold text-slate-900">
            Algorithmic Invariants
          </h2>
          <span className="text-xs font-mono tabular-nums font-medium text-[#059669]">
            {passCount}/{snapshot.invariants.length} PASS
          </span>
        </div>

        <div className="divide-y divide-slate-100">
          {snapshot.invariants.map((inv) => (
            <div key={inv.id} className="py-2.5 first:pt-0 last:pb-0 space-y-1">
              <div className="flex items-start justify-between gap-2">
                <span className="text-xs font-medium text-slate-900 leading-snug">
                  {inv.name}
                </span>
                <span
                  className={`inline-flex items-center gap-1 text-[11px] font-mono font-medium shrink-0 ${
                    inv.passed ? 'text-[#059669]' : 'text-[#DC2626]'
                  }`}
                >
                  {inv.passed ? (
                    <CheckCircle2 className="w-3.5 h-3.5" />
                  ) : (
                    <AlertTriangle className="w-3.5 h-3.5" />
                  )}
                  {inv.passed ? 'PASS' : 'FAIL'}
                </span>
              </div>
              <div className="text-[11px] font-mono text-slate-500 truncate">
                {inv.metricValue}
              </div>
            </div>
          ))}
        </div>
      </section>
    </aside>
  );
}
