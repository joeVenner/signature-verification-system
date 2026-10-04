import {
  type DesignSystem,
  type Page,
  type SlideMeta,
  type SlideTransition,
  Step,
  Steps,
  useIsActivePage,
  useSlidePageNumber,
} from '@open-slide/core';
import { type CSSProperties, type ReactNode, useEffect, useState } from 'react';

// Vault noir: near-black ink, one brass accent, editorial serif + mono numerals.
export const design: DesignSystem = {
  palette: { bg: '#0b0b0d', text: '#f2efe8', accent: '#c9a45c' },
  fonts: {
    display: '"Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif',
    body: '-apple-system, BlinkMacSystemFont, "Helvetica Neue", system-ui, sans-serif',
  },
  typeScale: { hero: 112, body: 34 },
  radius: 14,
};

const MONO = '"SF Mono", "JetBrains Mono", Menlo, Consolas, monospace';
const muted = '#8d8a83';
const dim = '#55534f';
const panel = '#141417';
const rule = '#26252a';
const good = '#7fc8a0';
const bad = '#e0796b';
const PAD = 120;

// Every value on these pages comes from benchmark/experiments.md (EXP-015..024) and the
// frozen clearance score in benchmark/clearance_score.py. Nothing is rounded up.
const SCORE = {
  valBefore: 3.88,
  valAfter: 5.24,
  testBefore: 5.26,
  testAfter: 6.0,
  holdBefore: 4.93 as number | null,
  holdAfter: 5.78 as number | null,
};

const KEYFRAMES = `
@keyframes sv-draw { from { stroke-dashoffset: 1; } to { stroke-dashoffset: 0; } }
@keyframes sv-rise { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: translateY(0); } }
@keyframes sv-fade { from { opacity: 0; } to { opacity: 1; } }
@keyframes sv-grow { from { transform: scaleX(0); } to { transform: scaleX(1); } }
@keyframes sv-pop { 0% { opacity: 0; transform: scale(0.4); } 70% { opacity: 1; transform: scale(1.15); } 100% { opacity: 1; transform: scale(1); } }
@keyframes sv-travel { from { left: 0%; } to { left: 100%; } }
@keyframes sv-glow { 0%, 100% { box-shadow: 0 0 0 0 rgba(201,164,92,0); } 50% { box-shadow: 0 0 36px 4px rgba(201,164,92,0.35); } }
@keyframes sv-sheen { from { background-position: -200% 0; } to { background-position: 200% 0; } }
`;

const Keyframes = () => <style>{KEYFRAMES}</style>;

const fill: CSSProperties = {
  width: '100%',
  height: '100%',
  background: 'var(--osd-bg)',
  color: 'var(--osd-text)',
  fontFamily: 'var(--osd-font-body)',
  position: 'relative',
  boxSizing: 'border-box',
};

/** Animation style only on the page the audience is watching; thumbnails render settled. */
const useAnim = () => {
  const active = useIsActivePage();
  return (animation: string): CSSProperties => (active ? { animation } : {});
};

/** Counts from `from` to `to` once the page is active; settled value elsewhere. */
const useCount = (from: number, to: number, ms: number, delay = 0) => {
  const active = useIsActivePage();
  const [value, setValue] = useState(active ? from : to);
  useEffect(() => {
    if (!active) return;
    let raf = 0;
    const start = performance.now() + delay;
    const tick = (now: number) => {
      const t = Math.min(1, Math.max(0, (now - start) / ms));
      const eased = 1 - Math.pow(1 - t, 3);
      setValue(from + (to - from) * eased);
      if (t < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [active, from, to, ms, delay]);
  return value;
};

const Eyebrow = ({ children }: { children: ReactNode }) => (
  <div style={{ fontFamily: MONO, fontSize: 24, letterSpacing: '0.28em', color: 'var(--osd-accent)', textTransform: 'uppercase' }}>
    {children}
  </div>
);

const Heading = ({ children }: { children: ReactNode }) => (
  <h2 style={{ fontFamily: 'var(--osd-font-display)', fontSize: 72, fontWeight: 600, margin: '20px 0 0', lineHeight: 1.15, letterSpacing: '-0.01em' }}>
    {children}
  </h2>
);

const Footer = () => {
  const { current, total } = useSlidePageNumber();
  return (
    <div style={{ position: 'absolute', left: PAD, right: PAD, bottom: 56, display: 'flex', justifyContent: 'space-between', fontFamily: MONO, fontSize: 20, color: dim, letterSpacing: '0.12em' }}>
      <span>DETERMINISTIC SIGNATURE VERIFICATION</span>
      <span>{String(current).padStart(2, '0')} / {String(total).padStart(2, '0')}</span>
    </div>
  );
};

// A made-up flourish (not a real person's signature), drawn stroke by stroke.
const SIGNATURE_PATH =
  'M20 170 C 60 60, 120 40, 130 120 S 90 220, 150 180 S 230 60, 260 110 S 250 200, 300 170 ' +
  'C 330 150, 340 90, 370 100 S 380 190, 420 160 C 450 140, 470 80, 500 95 S 510 175, 560 150 ' +
  'C 600 130, 640 70, 680 90 S 690 170, 740 140 C 780 120, 830 110, 880 125';

const Signature = ({ width, delay = 0, color = 'var(--osd-accent)', stroke = 6 }: { width: number; delay?: number; color?: string; stroke?: number }) => {
  const anim = useAnim();
  return (
    <svg viewBox="0 0 900 240" width={width} height={(width * 240) / 900} style={{ display: 'block', overflow: 'visible' }}>
      <path
        d={SIGNATURE_PATH}
        pathLength={1}
        fill="none"
        stroke={color}
        strokeWidth={stroke}
        strokeLinecap="round"
        strokeLinejoin="round"
        style={{ strokeDasharray: 1, strokeDashoffset: 0, ...anim(`sv-draw 2.4s cubic-bezier(.6,.05,.3,1) ${delay}s both`) }}
      />
    </svg>
  );
};

// ---------------------------------------------------------------------------
// 01 Cover
// ---------------------------------------------------------------------------

const Cover: Page = () => {
  const anim = useAnim();
  return (
    <div style={{ ...fill, display: 'flex', flexDirection: 'column', justifyContent: 'center', padding: `0 ${PAD + 40}px` }}>
      <Keyframes />
      <div style={{ position: 'absolute', inset: 0, background: 'radial-gradient(1200px 600px at 75% 30%, rgba(201,164,92,0.10), transparent 70%)' }} />
      <div style={anim('sv-fade 0.8s ease-out both')}>
        <Eyebrow>CPU only · No training · Same answer every time</Eyebrow>
      </div>
      <div style={{ margin: '56px 0 40px' }}>
        <Signature width={860} delay={0.3} />
      </div>
      <h1 style={{ fontFamily: 'var(--osd-font-display)', fontSize: 'var(--osd-size-hero)', fontWeight: 600, margin: 0, lineHeight: 1.05, letterSpacing: '-0.02em', ...anim('sv-rise 0.9s ease-out 1.6s both') }}>
        Same signature. Same answer.
      </h1>
      <p style={{ fontSize: 40, color: muted, margin: '36px 0 0', ...anim('sv-rise 0.9s ease-out 2s both') }}>
        A deterministic signature verifier for banking — and an honest scorecard.
      </p>
    </div>
  );
};

// ---------------------------------------------------------------------------
// 02 The problem
// ---------------------------------------------------------------------------

const InputCard = ({ label, sub, delay }: { label: string; sub: string; delay: number }) => {
  const anim = useAnim();
  return (
    <div style={{ width: 460, height: 300, background: panel, border: `1px solid ${rule}`, borderRadius: 'var(--osd-radius)', padding: 36, boxSizing: 'border-box', display: 'flex', flexDirection: 'column', justifyContent: 'space-between', ...anim(`sv-rise 0.7s ease-out ${delay}s both`) }}>
      <div style={{ fontFamily: MONO, fontSize: 22, color: muted, letterSpacing: '0.18em' }}>{label}</div>
      <Signature width={380} delay={delay + 0.3} color="var(--osd-text)" stroke={5} />
      <div style={{ fontSize: 28, color: muted }}>{sub}</div>
    </div>
  );
};

const Output = ({ name, value, delay }: { name: string; value: string; delay: number }) => {
  const anim = useAnim();
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', borderBottom: `1px solid ${rule}`, padding: '18px 0', ...anim(`sv-rise 0.6s ease-out ${delay}s both`) }}>
      <span style={{ fontSize: 34 }}>{name}</span>
      <span style={{ fontFamily: MONO, fontSize: 26, color: 'var(--osd-accent)' }}>{value}</span>
    </div>
  );
};

const Chip = ({ children }: { children: ReactNode }) => (
  <span style={{ fontFamily: MONO, fontSize: 22, letterSpacing: '0.1em', color: 'var(--osd-text)', border: `1px solid ${rule}`, borderRadius: 999, padding: '12px 26px' }}>{children}</span>
);

const Problem: Page = () => (
  <div style={{ ...fill, padding: PAD }}>
    <Keyframes />
    <Eyebrow>The job</Eyebrow>
    <Heading>Two images in. One decision out.</Heading>
    <div style={{ display: 'flex', alignItems: 'center', gap: 40, marginTop: 64 }}>
      <InputCard label="A · REFERENCE ON FILE" sub="Specimen card" delay={0.1} />
      <InputCard label="B · TO VERIFY" sub="Cheque / form" delay={0.4} />
      <div style={{ flex: 1, marginLeft: 20 }}>
        <Output name="Similarity" value="0–100" delay={1.2} />
        <Output name="Confidence" value="distance to threshold" delay={1.4} />
        <Output name="Decision" value="MATCH · REVIEW · NO MATCH" delay={1.6} />
      </div>
    </div>
    <div style={{ display: 'flex', gap: 20, marginTop: 72 }}>
      <Chip>CPU ONLY</Chip>
      <Chip>NO TRAINING</Chip>
      <Chip>NO RANDOMNESS</Chip>
      <Chip>EXPLAINABLE</Chip>
    </div>
    <Footer />
  </div>
);

// ---------------------------------------------------------------------------
// 03 Pipeline
// ---------------------------------------------------------------------------

const Stage = ({ n, title, sub, delay, last = false }: { n: string; title: string; sub: string; delay: number; last?: boolean }) => {
  const anim = useAnim();
  return (
    <div style={{ width: 196, ...anim(`sv-rise 0.6s ease-out ${delay}s both`) }}>
      <div style={{ width: 64, height: 64, borderRadius: 999, border: `2px solid ${last ? 'var(--osd-accent)' : rule}`, background: last ? 'rgba(201,164,92,0.12)' : panel, display: 'flex', alignItems: 'center', justifyContent: 'center', fontFamily: MONO, fontSize: 22, color: 'var(--osd-accent)', position: 'relative', zIndex: 2 }}>
        {n}
      </div>
      <div style={{ fontSize: 30, fontWeight: 600, marginTop: 28, lineHeight: 1.2 }}>{title}</div>
      <div style={{ fontSize: 24, color: muted, marginTop: 12, lineHeight: 1.4 }}>{sub}</div>
    </div>
  );
};

const Pipeline: Page = () => {
  const anim = useAnim();
  return (
    <div style={{ ...fill, padding: PAD }}>
      <Keyframes />
      <Eyebrow>How it works</Eyebrow>
      <Heading>Eight steps, all plain arithmetic.</Heading>
      <div style={{ position: 'relative', marginTop: 96 }}>
        <div style={{ position: 'absolute', top: 31, left: 32, right: 140, height: 2, background: rule }} />
        <div style={{ position: 'absolute', top: 31, left: 32, right: 140, height: 2, background: 'var(--osd-accent)', transformOrigin: 'left', ...anim('sv-grow 3.2s cubic-bezier(.5,0,.2,1) 0.3s both') }} />
        <div style={{ position: 'absolute', top: 24, left: 32, right: 140, height: 16 }}>
          <div style={{ position: 'absolute', width: 16, height: 16, borderRadius: 999, background: 'var(--osd-accent)', marginLeft: -8, boxShadow: '0 0 24px 6px rgba(201,164,92,0.55)', ...anim('sv-travel 3.2s cubic-bezier(.5,0,.2,1) 0.3s infinite') }} />
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
          <Stage n="01" title="Clean the page" sub="Paper, shadows, tint removed" delay={0.2} />
          <Stage n="02" title="Isolate the ink" sub="Print & ruled lines dropped" delay={0.5} />
          <Stage n="03" title="Normalise" sub="Crop, flat-field, denoise" delay={0.8} />
          <Stage n="04" title="Trace strokes" sub="Skeleton + pen width" delay={1.1} />
          <Stage n="05" title="Align" sub="ICP, both directions" delay={1.4} />
          <Stage n="06" title="Measure" sub="8 similarity signals" delay={1.7} />
          <Stage n="07" title="Fuse" sub="Fixed logistic weights" delay={2.0} />
          <Stage n="08" title="Decide" sub="Validated thresholds" delay={2.3} last />
        </div>
      </div>
      <p style={{ fontSize: 34, color: muted, marginTop: 96, ...anim('sv-fade 0.8s ease-out 2.8s both') }}>
        ≈ 0.4 s per comparison on one CPU core · every step is shown live in the console.
      </p>
      <Footer />
    </div>
  );
};

// ---------------------------------------------------------------------------
// 04 Signals
// ---------------------------------------------------------------------------

const HeroSignal = ({ name, what, auc, delay }: { name: string; what: string; auc: string; delay: number }) => {
  const anim = useAnim();
  return (
    <div style={{ flex: 1, background: panel, border: `1px solid rgba(201,164,92,0.45)`, borderRadius: 'var(--osd-radius)', padding: 44, ...anim(`sv-rise 0.7s ease-out ${delay}s both, sv-glow 3.5s ease-in-out ${delay + 0.8}s infinite`) }}>
      <div style={{ fontFamily: 'var(--osd-font-display)', fontSize: 52, fontWeight: 600 }}>{name}</div>
      <div style={{ fontSize: 30, color: muted, marginTop: 16 }}>{what}</div>
      <div style={{ fontFamily: MONO, fontSize: 26, color: 'var(--osd-accent)', marginTop: 32 }}>{auc}</div>
    </div>
  );
};

const SmallSignal = ({ name }: { name: string }) => (
  <div style={{ flex: 1, fontSize: 26, color: muted, border: `1px solid ${rule}`, borderRadius: 'var(--osd-radius)', padding: '22px 20px', textAlign: 'center' }}>{name}</div>
);

const Signals: Page = () => (
  <div style={{ ...fill, padding: PAD }}>
    <Keyframes />
    <Eyebrow>What it compares</Eyebrow>
    <Heading>Forgers copy the shape. Not the hand.</Heading>
    <div style={{ display: 'flex', gap: 40, marginTop: 64 }}>
      <HeroSignal name="Stroke direction" what="Do matched strokes travel the same way?" auc="single-signal AUC 0.937" delay={0.2} />
      <HeroSignal name="Pressure pattern" what="Is the ink darkest in the same places?" auc="single-signal AUC 0.907" delay={0.5} />
    </div>
    <div style={{ fontFamily: MONO, fontSize: 22, color: dim, letterSpacing: '0.18em', marginTop: 64 }}>PLUS SIX SUPPORTING SIGNALS</div>
    <div style={{ display: 'flex', gap: 20, marginTop: 24 }}>
      <SmallSignal name="Keypoints" />
      <SmallSignal name="Slant" />
      <SmallSignal name="Row rhythm" />
      <SmallSignal name="Column rhythm" />
      <SmallSignal name="Pen width" />
      <SmallSignal name="Curvature" />
    </div>
    <Footer />
  </div>
);

// ---------------------------------------------------------------------------
// 05 Determinism
// ---------------------------------------------------------------------------

const DotRow = ({ row }: { row: number }) => {
  const anim = useAnim();
  const dots: ReactNode[] = [];
  for (let i = 0; i < 20; i += 1) {
    const k = row * 20 + i;
    dots.push(
      <div key={k} style={{ width: 34, height: 34, borderRadius: 999, background: 'var(--osd-accent)', ...anim(`sv-pop 0.35s ease-out ${0.3 + k * 0.022}s both`) }} />,
    );
  }
  return <div style={{ display: 'flex', gap: 22 }}>{dots}</div>;
};

const Determinism: Page = () => {
  const anim = useAnim();
  const runs = Math.round(useCount(0, 100, 2200, 300));
  return (
    <div style={{ ...fill, padding: PAD }}>
      <Keyframes />
      <Eyebrow>Non-negotiable</Eyebrow>
      <Heading>{runs} runs. One answer.</Heading>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 22, marginTop: 72 }}>
        <DotRow row={0} />
        <DotRow row={1} />
        <DotRow row={2} />
        <DotRow row={3} />
        <DotRow row={4} />
      </div>
      <p style={{ fontSize: 34, color: muted, marginTop: 72, lineHeight: 1.5, ...anim('sv-fade 0.8s ease-out 2.8s both') }}>
        Byte-identical JSON on every run, and across fresh processes. Enforced by a test in CI.
      </p>
      <Footer />
    </div>
  );
};

// ---------------------------------------------------------------------------
// 06 The score
// ---------------------------------------------------------------------------

const ScorePair = ({ label, before, after, note }: { label: string; before: number | null; after: number | null; note: string }) => (
  <div style={{ flex: 1, borderTop: `1px solid ${rule}`, paddingTop: 28 }}>
    <div style={{ fontFamily: MONO, fontSize: 22, color: muted, letterSpacing: '0.16em' }}>{label}</div>
    <div style={{ fontFamily: MONO, fontSize: 52, marginTop: 14 }}>
      {before === null || after === null ? (
        <span style={{ color: dim }}>pending</span>
      ) : (
        <>
          <span style={{ color: muted }}>{before.toFixed(2)}</span>
          <span style={{ color: dim }}> → </span>
          <span style={{ color: 'var(--osd-accent)' }}>{after.toFixed(2)}</span>
        </>
      )}
    </div>
    <div style={{ fontSize: 26, color: muted, marginTop: 10 }}>{note}</div>
  </div>
);

const Score: Page = () => {
  const anim = useAnim();
  const value = useCount(SCORE.valBefore, SCORE.valAfter, 2400, 400);
  return (
    <div style={{ ...fill, padding: PAD }}>
      <Keyframes />
      <Eyebrow>Clearance score · 0–10 · frozen before we started</Eyebrow>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 40, marginTop: 40 }}>
        <div style={{ fontFamily: MONO, fontSize: 300, lineHeight: 1, letterSpacing: '-0.04em', background: 'linear-gradient(100deg, #f2efe8 30%, #c9a45c 50%, #f2efe8 70%)', backgroundSize: '200% 100%', WebkitBackgroundClip: 'text', backgroundClip: 'text', color: 'transparent', ...anim('sv-sheen 3s ease-in-out 2.8s 1 both') }}>
          {value.toFixed(2)}
        </div>
        <div style={{ fontSize: 44, color: muted }}>from {SCORE.valBefore.toFixed(2)}</div>
      </div>
      <div style={{ display: 'flex', gap: 64, marginTop: 72, ...anim('sv-rise 0.8s ease-out 2.6s both') }}>
        <ScorePair label="VALIDATION" before={SCORE.valBefore} after={SCORE.valAfter} note="decided every keep / revert" />
        <ScorePair label="TEST" before={SCORE.testBefore} after={SCORE.testAfter} note="never tuned on" />
        <ScorePair label="HOLDOUT" before={SCORE.holdBefore} after={SCORE.holdAfter} note="scored exactly once" />
      </div>
      <Footer />
    </div>
  );
};

// ---------------------------------------------------------------------------
// 07 Benchmarks
// ---------------------------------------------------------------------------

/** One before/after metric; bars share a 0–35% scale so lengths are comparable. */
const Metric = ({ label, before, after, lowerIsBetter, delay }: { label: string; before: number; after: number; lowerIsBetter: boolean; delay: number }) => {
  const anim = useAnim();
  const scale = 760 / 35;   // widest bar (30.9%) stays inside the ~790 px bar column
  const better = lowerIsBetter ? after < before : after > before;
  const same = after === before;
  const color = same ? muted : better ? good : bad;
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '520px 1fr 300px', alignItems: 'center', gap: 32 }}>
      <div style={{ fontSize: 30 }}>{label}</div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
        <div style={{ height: 14, width: Math.max(4, before * scale), background: dim, borderRadius: 7, transformOrigin: 'left', ...anim(`sv-grow 0.9s ease-out ${delay}s both`) }} />
        <div style={{ height: 14, width: Math.max(4, after * scale), background: color, borderRadius: 7, transformOrigin: 'left', ...anim(`sv-grow 0.9s ease-out ${delay + 0.25}s both`) }} />
      </div>
      <div style={{ fontFamily: MONO, fontSize: 30, textAlign: 'right' }}>
        <span style={{ color: muted }}>{before.toFixed(1)}%</span>
        <span style={{ color: dim }}> → </span>
        <span style={{ color }}>{after.toFixed(1)}%</span>
      </div>
    </div>
  );
};

const Benchmarks: Page = () => (
  <div style={{ ...fill, padding: PAD }}>
    <Keyframes />
    <Eyebrow>Untouched test images · before → after</Eyebrow>
    <Heading>Where it got better. And where it didn't.</Heading>
    <div style={{ display: 'flex', flexDirection: 'column', gap: 44, marginTop: 72 }}>
      <Metric label="Printed text next to signature (EER)" before={30.9} after={8.1} lowerIsBetter delay={0.2} />
      <Metric label="Rotated 20° (EER)" before={18.3} after={13.2} lowerIsBetter delay={0.5} />
      <Metric label="Clean forgery error (EER)" before={9.3} after={9.0} lowerIsBetter delay={0.8} />
      <Metric label="Genuine auto-matched" before={15.9} after={33.2} lowerIsBetter={false} delay={1.1} />
      <Metric label="Half-size image (EER)" before={15.9} after={17.5} lowerIsBetter delay={1.4} />
    </div>
    <p style={{ fontSize: 28, color: muted, marginTop: 56 }}>Forgeries auto-matched: 0% before, 0% after. Grey = before, colour = after.</p>
    <Footer />
  </div>
);

// ---------------------------------------------------------------------------
// 08 What failed
// ---------------------------------------------------------------------------

const Rejected = ({ what, tried, result }: { what: string; tried: string; result: string }) => (
  <div style={{ display: 'grid', gridTemplateColumns: '640px 260px 1fr', alignItems: 'baseline', borderBottom: `1px solid ${rule}`, padding: '22px 0' }}>
    <span style={{ fontSize: 34 }}>{what}</span>
    <span style={{ fontFamily: MONO, fontSize: 24, color: muted }}>{tried}</span>
    <span style={{ fontSize: 28, color: bad }}>{result}</span>
  </div>
);

const Failed: Page = () => (
  <div style={{ ...fill, padding: PAD }}>
    <Keyframes />
    <Eyebrow>Tested, measured, rejected</Eyebrow>
    <Heading>160+ ideas that didn't make the cut.</Heading>
    <div style={{ marginTop: 48 }}>
      <Steps>
        <Step duration={260}><Rejected what="Elastic deformation signals" tried="≈120 configs" result="≤ 0.3 pt gain — noise" /></Step>
        <Step duration={260}><Rejected what="Local stroke patterns" tried="15 configs" result="0.25 pt — redundant" /></Step>
        <Step duration={260}><Rejected what="Re-tuning image clean-up" tried="17 settings" result="all within noise" /></Step>
        <Step duration={260}><Rejected what="SigNet pretrained network" tried="10 configs" result="+0.03 pt · non-commercial" /></Step>
        <Step duration={260}><Rejected what="Scale + rotation branch" tried="1 branch" result="score 5.24 → 5.17" /></Step>
      </Steps>
    </div>
    <p style={{ fontSize: 30, color: muted, marginTop: 48 }}>A gain only counted if it survived resampling the writers.</p>
    <Footer />
  </div>
);

// ---------------------------------------------------------------------------
// 09 Honest limits
// ---------------------------------------------------------------------------

const Limit = ({ big, text }: { big: string; text: string }) => (
  <div style={{ display: 'flex', alignItems: 'baseline', gap: 40, padding: '26px 0', borderBottom: `1px solid ${rule}` }}>
    <span style={{ fontFamily: MONO, fontSize: 48, color: 'var(--osd-accent)', width: 300, flexShrink: 0 }}>{big}</span>
    <span style={{ fontSize: 34 }}>{text}</span>
  </div>
);

const Limits: Page = () => (
  <div style={{ ...fill, padding: PAD }}>
    <Keyframes />
    <Eyebrow>Read this before relying on it</Eyebrow>
    <Heading>What it can't do yet.</Heading>
    <div style={{ marginTop: 56 }}>
      <Steps>
        <Step duration={260}><Limit big="7 / 10" text="Target not reached. Plateau is about 5.5–6 with these constraints." /></Step>
        <Step duration={260}><Limit big="2 in 3" text="Genuine signatures still go to manual review with one reference." /></Step>
        <Step duration={260}><Limit big="9–12%" text="Forgery error rate (EER) on clean scans, depending on the image set." /></Step>
        <Step duration={260}><Limit big="Lab scans" text="Benchmarked on CEDAR, not on real counter photos or cheques." /></Step>
      </Steps>
    </div>
    <Footer />
  </div>
);

// ---------------------------------------------------------------------------
// 10 Next
// ---------------------------------------------------------------------------

const NextCard = ({ rank, title, text, delay }: { rank: string; title: string; text: string; delay: number }) => {
  const anim = useAnim();
  return (
    <div style={{ flex: 1, background: panel, border: `1px solid ${rule}`, borderRadius: 'var(--osd-radius)', padding: 44, ...anim(`sv-rise 0.7s ease-out ${delay}s both`) }}>
      <div style={{ fontFamily: MONO, fontSize: 24, color: 'var(--osd-accent)' }}>{rank}</div>
      <div style={{ fontFamily: 'var(--osd-font-display)', fontSize: 46, fontWeight: 600, marginTop: 20, lineHeight: 1.15 }}>{title}</div>
      <div style={{ fontSize: 28, color: muted, marginTop: 20, lineHeight: 1.5 }}>{text}</div>
    </div>
  );
};

const Next: Page = () => (
  <div style={{ ...fill, padding: PAD }}>
    <Keyframes />
    <Eyebrow>What moves the score next</Eyebrow>
    <Heading>Bigger levers than more tuning.</Heading>
    <div style={{ display: 'flex', gap: 40, marginTop: 72 }}>
      <NextCard rank="01 · BIGGEST" title="Three references, not one" text="Built in already. In earlier tests it cut forgery error by about a third." delay={0.2} />
      <NextCard rank="02" title="Real bank images" text="Tune on the failures that actually happen at the counter." delay={0.5} />
      <NextCard rank="03" title="A licensed model" text="A commercially licensed signature network, frozen, on CPU." delay={0.8} />
    </div>
    <p style={{ fontFamily: MONO, fontSize: 26, color: muted, marginTop: 80 }}>Try it live: python -m signature_verification_system.serve → localhost:8765</p>
    <Footer />
  </div>
);

// ---------------------------------------------------------------------------
// Transitions: one quiet house cut; the cover settles in.
// ---------------------------------------------------------------------------

const EASE_OUT = 'cubic-bezier(0, 0, 0.2, 1)';
const EASE_IN = 'cubic-bezier(0.4, 0, 1, 1)';
const HOLD: Keyframe[] = [{ opacity: 1 }, { opacity: 1 }];

export const transition: SlideTransition = {
  duration: 260,
  exit: { duration: 260, easing: EASE_IN, keyframes: HOLD },
  enter: {
    duration: 260,
    easing: EASE_OUT,
    keyframes: [
      { opacity: 0, transform: 'translateY(6px)' },
      { opacity: 1, transform: 'translateY(0)' },
    ],
  },
};

Cover.transition = {
  duration: 280,
  exit: { duration: 280, easing: EASE_IN, keyframes: HOLD },
  enter: {
    duration: 280,
    easing: EASE_OUT,
    keyframes: [
      { opacity: 0, transform: 'translateY(12px)', filter: 'blur(4px)' },
      { opacity: 1, transform: 'translateY(0)', filter: 'blur(0)' },
    ],
  },
};

export const notes: (string | undefined)[] = [
  'Open: a bank needs the same answer for the same two images, every time. That is the whole design brief.',
  'Two inputs: the specimen on file and the signature to check. Three outputs. Hard constraints: CPU only, no training, no randomness.',
  'Walk the pipeline left to right. Stress that every step is classical image processing and fixed arithmetic; the console shows each stage on real images.',
  'The two signals that carry the system: stroke direction after alignment, and where the pen pressed. Forgers reproduce the outline, not these.',
  'Determinism is tested, not promised: 100 identical runs plus fresh processes, byte-for-byte.',
  'The score was frozen before any work started. Validation drove decisions, so it is the optimistic number; test and holdout are the honest ones.',
  'Green got better, red got worse. Half-size images regressed slightly; the fix exists on a branch but cost clean accuracy.',
  'We report what failed too. Over 160 configurations rejected because the gain did not survive resampling the writers.',
  'Be direct: 7/10 was not reached. With one reference, most genuine signatures still need a human.',
  'The levers that matter now are data and references, not more tuning. Invite people to try the live console.',
];

export const meta: SlideMeta = {
  title: 'Deterministic Signature Verification',
  createdAt: '2026-10-04T09:37:51.461Z',
};

export default [Cover, Problem, Pipeline, Signals, Determinism, Score, Benchmarks, Failed, Limits, Next] satisfies Page[];
