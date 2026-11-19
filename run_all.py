#!/usr/bin/env python3
"""
A6P3 - Predictive process mining for robotic work-order logs.
Runs EVERYTHING: preprocessing, prefix extraction at checkpoints, proposed TCN, baselines
(process-model average, LightGBM, MLP), ablation, sensitivity, temporal hold-out,
significance tests, log-based bottleneck mining + what-if bound, and all figures.

Usage:  python3 run_all.py            (from anywhere; reads ../data, writes ../results ../figures)
        QUICK=1 python3 run_all.py    (2 folds, few epochs - smoke test only)
"""
import os, sys, json, time, math, pickle, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import torch, torch.nn as nn, torch.nn.functional as F
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score, precision_recall_curve
from sklearn.tree import DecisionTreeRegressor
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data'); RES = os.path.join(HERE, '..', 'results'); FIG = os.path.join(HERE, '..', 'figures')
os.makedirs(RES, exist_ok=True); os.makedirs(FIG, exist_ok=True)
QUICK = os.environ.get('QUICK') == '1'
FIGS_ONLY = os.environ.get('FIGS_ONLY') == '1'   # redraw figures from results/raw_checkpoint.pkl
SEED = 2026
NFOLD = 2 if QUICK else 5
MAX_EP = 6 if QUICK else 60
RHO_TRAIN = [0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9]
RHO_EVAL = [0.25, 0.5, 0.75]
HOT_DEPTH = int(os.environ.get('HOT_DEPTH', 4))
T0 = time.time()
def log(*a):
    print(f'[{time.time()-T0:7.1f}s]', *a, flush=True)

plt.rcParams.update({'font.size': 7.5, 'font.family': 'serif', 'axes.titlesize': 7.5, 'axes.labelsize': 7.5,
                     'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7, 'figure.dpi': 100})
COL = {'TCN (proposed)': '#c0392b', 'LightGBM': '#2c7fb8', 'MLP': '#7fa650', 'Process-model avg.': '#8c8c8c'}

def savefig(fig, name):
    fig.savefig(os.path.join(FIG, name + '.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(RES, name + '.jpg'), dpi=300, bbox_inches='tight', pil_kwargs={'quality': 92})
    plt.close(fig)

# =====================================================================================
# 1. Load + preprocessing (imputation, outlier repair)
# =====================================================================================
cases = pd.read_csv(os.path.join(DATA, 'A6P3_workorder_cases.csv'), parse_dates=['release_timestamp'])
ev = pd.read_csv(os.path.join(DATA, 'A6P3_workorder_event_log.csv'), parse_dates=['start_timestamp', 'complete_timestamp'])
ev = ev.sort_values(['case_id', 'start_timestamp', 'event_id']).reset_index(drop=True)
cases = cases.sort_values('release_timestamp').reset_index(drop=True)
EPOCH = cases.release_timestamp.min().normalize()
def hrs(ts): return (ts - EPOCH).dt.total_seconds().values / 3600.0

prep = {}
# -- event-level outliers: logged duration / waiting vs. the timestamps (redundant source)
ev = ev.merge(cases[['case_id', 'release_timestamp', 'production_cell']], on='case_id', how='left')
dur_ts = (ev.complete_timestamp - ev.start_timestamp).dt.total_seconds() / 60
prev_c = ev.groupby('case_id').complete_timestamp.shift()
prev_c = prev_c.fillna(ev.release_timestamp)
gap_ts = (ev.start_timestamp - prev_c).dt.total_seconds() / 60
bad_d = (ev.duration_min - dur_ts).abs() > 0.5 * dur_ts + 5
bad_w = (ev.waiting_min - gap_ts).abs() > 0.5 * gap_ts.abs() + 5
ev.loc[bad_d, 'duration_min'] = dur_ts[bad_d]; ev.loc[bad_w, 'waiting_min'] = gap_ts[bad_w].clip(lower=0)
prep['duration_outliers_repaired'] = int(bad_d.sum()); prep['waiting_outliers_repaired'] = int(bad_w.sum())
# -- missing resource: mode of (activity, cell)
miss_r = ev.resource.isna()
mode_r = ev.dropna(subset=['resource']).groupby(['activity', 'production_cell']).resource.agg(lambda s: s.mode().iloc[0])
ev.loc[miss_r, 'resource'] = [mode_r.get((a, c), 'UNK') for a, c in zip(ev.loc[miss_r, 'activity'], ev.loc[miss_r, 'production_cell'])]
prep['resource_imputed'] = int(miss_r.sum())
# -- missing plant_load: within-case interpolation, then median
miss_l = ev.plant_load.isna(); prep['plant_load_imputed'] = int(miss_l.sum())
ev['plant_load'] = ev.groupby('case_id').plant_load.transform(lambda s: s.interpolate(limit_direction='both'))
ev['plant_load'] = ev.plant_load.fillna(ev.plant_load.median())
# -- case-level: complexity / quantity median per family + missing flag
for col in ['complexity_index', 'order_quantity']:
    cases[col + '_miss'] = cases[col].isna().astype(int)
    prep[col + '_imputed'] = int(cases[col].isna().sum())
    cases[col] = cases[col].fillna(cases.groupby('product_family')[col].transform('median'))
log('preprocessing', prep)

ACTS = sorted(ev.activity.unique()); RESS = sorted(ev.resource.unique())
A2I = {a: i + 1 for i, a in enumerate(ACTS)}; R2I = {r: i + 1 for i, r in enumerate(RESS)}  # 0 = pad
FAMS = sorted(cases.product_family.unique()); CELLS = sorted(cases.production_cell.unique())
ev['s_h'] = hrs(ev.start_timestamp); ev['c_h'] = hrs(ev.complete_timestamp)
cases['rel_h'] = hrs(cases.release_timestamp); cases['end_h'] = cases.rel_h + cases.cycle_time_hours
REL = cases.rel_h.values; END = cases.end_h.values
rel_sorted = np.sort(REL); end_sorted = np.sort(END)
cell_rel = {c: np.sort(cases.rel_h[cases.production_cell == c].values) for c in CELLS}
cell_end = {c: np.sort(cases.end_h[cases.production_cell == c].values) for c in CELLS}
def wip(t, rs, es): return np.searchsorted(rs, t, 'right') - np.searchsorted(es, t, 'right')

# =====================================================================================
# 2. Prefix extraction at checkpoints tau = rho * due (no outcome columns used)
# =====================================================================================
LMAX = 16
TOK_NUM = ['dur', 'wait', 'fault', 'load', 'offset', 'hsin', 'hcos', 'inprog', 'inact']
evg = {k: g for k, g in ev.groupby('case_id')}
STATIC_COLS = (['qty', 'cplx', 'rush', 'short', 'due', 'qty_miss', 'cplx_miss'] + [f'fam_{f}' for f in FAMS] + [f'cell_{c}' for c in CELLS])
CTX_COLS = ['rho', 'tau', 'slack', 'since_last', 'tod_sin', 'tod_cos', 'dow_sat', 'dow_sun', 'wip_plant', 'wip_cell']

def build_prefixes(rhos):
    rows, seqA, seqR, seqN = [], [], [], []
    for ci, cr in cases.iterrows():
        g = evg[cr.case_id]; rel = cr.rel_h; due = cr.due_in_hours; ct = cr.cycle_time_hours
        s_h = g.s_h.values - rel; c_h = g.c_h.values - rel
        acts = g.activity.values; resv = g.resource.values
        static = [cr.order_quantity / 100, cr.complexity_index, int(cr.priority == 'rush'), cr.material_shortage, due / 100,
                  cr.order_quantity_miss, cr.complexity_index_miss] + [int(cr.product_family == f) for f in FAMS] + \
                 [int(cr.production_cell == c) for c in CELLS]
        for rho in rhos:
            tau = rho * due
            if ct <= tau: continue  # case already finished -> nothing to predict
            done = c_h <= tau; inp = (s_h <= tau) & (c_h > tau)
            idx = list(np.where(done)[0]) + list(np.where(inp)[0])
            A = np.zeros(LMAX, int); R = np.zeros(LMAX, int); N = np.zeros((LMAX, len(TOK_NUM)), np.float32)
            idx = idx[-LMAX:]; off = LMAX - len(idx)
            for j, k in enumerate(idx):
                ip = bool(inp[k]); hod = ((rel + s_h[k]) % 24) / 24 * 2 * np.pi
                A[off + j] = A2I[acts[k]]; R[off + j] = R2I[resv[k]]
                N[off + j] = [0 if ip else np.log1p(g.duration_min.values[k] / 60), np.log1p(g.waiting_min.values[k] / 60),
                              0 if ip else g.robot_fault.values[k], g.plant_load.values[k], s_h[k] / 24,
                              np.sin(hod), np.cos(hod), float(ip), np.log1p(tau - s_h[k]) if ip else 0.0]
            last_c = c_h[done].max() if done.any() else 0.0
            tabs = rel + tau; tod = (tabs % 24) / 24 * 2 * np.pi; dow = int(((EPOCH + pd.Timedelta(hours=tabs)).dayofweek))
            ctx = [rho, tau / 24, (due - tau) / 24, np.log1p(tau - last_c), np.sin(tod), np.cos(tod), int(dow == 5), int(dow == 6),
                   wip(tabs, rel_sorted, end_sorted) / 30, wip(tabs, cell_rel[cr.production_cell], cell_end[cr.production_cell]) / 10]
            # aggregate (bag-of-activities) encoding for GBM / MLP
            agg = {}
            for a in ACTS:
                m = done & (acts == a); agg[f'n_{a}'] = m.sum()
                agg[f'w_{a}'] = np.log1p(g.waiting_min.values[m].sum() / 60)
            dn = np.where(done)[0]
            agg.update({'n_done': done.sum(), 'sum_dur': np.log1p(g.duration_min.values[done].sum() / 60),
                        'sum_wait': np.log1p(g.waiting_min.values[done].sum() / 60), 'n_fault': g.robot_fault.values[done].sum(),
                        'last_fault': g.robot_fault.values[dn[-1]] if len(dn) else 0,
                        'last_wait': np.log1p(g.waiting_min.values[dn[-1]] / 60) if len(dn) else 0,
                        'mean_load': g.plant_load.values[done].mean() if len(dn) else 0,
                        'inprog_elapsed': np.log1p(tau - s_h[inp].max()) if inp.any() else 0})
            lastA = acts[dn[-1]] if len(dn) else 'none'; inA = acts[inp][0] if inp.any() else 'none'
            inR = resv[inp][0] if inp.any() else 'none'
            for a in ACTS: agg[f'last_{a}'] = int(lastA == a); agg[f'cur_{a}'] = int(inA == a)
            for r in RESS: agg[f'curres_{r}'] = int(inR == r)
            row = dict(case_idx=ci, case_id=cr.case_id, rho=rho, rem=ct - tau, late=int(cr.late_completion),
                       state=(rho, tuple(acts[done]), inA), state2=(rho, lastA, inA))
            row.update({f's_{k}': v for k, v in zip(STATIC_COLS, static)}); row.update({f'c_{k}': v for k, v in zip(CTX_COLS, ctx)})
            row.update({f'a_{k}': v for k, v in agg.items()})
            rows.append(row); seqA.append(A); seqR.append(R); seqN.append(N)
    return pd.DataFrame(rows), np.stack(seqA), np.stack(seqR), np.stack(seqN)

P, SA, SR, SN = build_prefixes(RHO_TRAIN)
P['is_eval'] = P.rho.isin(RHO_EVAL)
S_COLS = [c for c in P.columns if c.startswith('s_')]; C_COLS = [c for c in P.columns if c.startswith('c_')]
AG_COLS = [c for c in P.columns if c.startswith('a_')]
log('prefixes', len(P), 'eval prefixes', int(P.is_eval.sum()), 'n features agg', len(AG_COLS) + len(S_COLS) + len(C_COLS))

# =====================================================================================
# 3. Models
# =====================================================================================
class CausalBlock(nn.Module):
    def __init__(self, ch, k, d, p):
        super().__init__()
        self.pad = (k - 1) * d
        self.c1 = nn.Conv1d(ch, ch, k, dilation=d); self.c2 = nn.Conv1d(ch, ch, k, dilation=d); self.dp = nn.Dropout(p)
    def forward(self, x):
        h = self.dp(F.relu(self.c1(F.pad(x, (self.pad, 0)))))
        h = self.dp(F.relu(self.c2(F.pad(h, (self.pad, 0)))))
        return F.relu(x + h)

class TCN(nn.Module):
    def __init__(self, n_act, n_res, n_num, n_side, ch=32, k=3, levels=3, dilate=True, p=0.1, use_attr=True, pool='last'):
        super().__init__()
        self.use_attr = use_attr; self.pool = pool
        self.ea = nn.Embedding(n_act + 1, 8, padding_idx=0); self.er = nn.Embedding(n_res + 1, 8, padding_idx=0)
        self.inp = nn.Conv1d(16 + (n_num if use_attr else 0), ch, 1)
        self.blocks = nn.Sequential(*[CausalBlock(ch, k, 2 ** i if dilate else 1, p) for i in range(levels)])
        self.side = nn.Sequential(nn.Linear(n_side, 32), nn.ReLU())
        self.trunk = nn.Sequential(nn.Linear(ch * (2 if pool == 'lastmax' else 1) + 32, 64), nn.ReLU(), nn.Dropout(p))
        self.reg = nn.Linear(64, 1); self.cls = nn.Linear(64, 1)
    def forward(self, A, R, N, S):
        x = [self.ea(A), self.er(R)] + ([N] if self.use_attr else [])
        x = torch.cat(x, -1).transpose(1, 2)
        hs = self.blocks(self.inp(x))
        h = hs[:, :, -1]
        if self.pool == 'lastmax':  # last state + max over valid time steps
            mask = (A > 0).unsqueeze(1)
            h = torch.cat([h, hs.masked_fill(~mask, -1e4).max(2).values], 1)
        z = self.trunk(torch.cat([h, self.side(S)], 1))
        return F.softplus(self.reg(z)).squeeze(1), self.cls(z).squeeze(1)

class MLP(nn.Module):
    def __init__(self, d, p=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Dropout(p), nn.Linear(128, 64), nn.ReLU(), nn.Dropout(p))
        self.reg = nn.Linear(64, 1); self.cls = nn.Linear(64, 1)
    def forward(self, X):
        z = self.net(X); return F.softplus(self.reg(z)).squeeze(1), self.cls(z).squeeze(1)

HP = dict(ch=16, k=3, levels=3, pool='last', side_agg=False, dropout=0.3, lr=1e-3, wd=1e-3, batch=256, lam=1.0, patience=8, scale_h=24.0, logt=False)
def fwd_t(y): return np.log1p(y / HP['scale_h']) if HP['logt'] else y / HP['scale_h']
def inv_t(z): return np.expm1(z) * HP['scale_h'] if HP['logt'] else z * HP['scale_h']

def train_torch(make, inputs_tr, y_tr, l_tr, inputs_va, y_va, l_va, seed, lam=1.0, curve=False):
    torch.manual_seed(seed); np.random.seed(seed)
    model = make(); opt = torch.optim.Adam(model.parameters(), lr=HP['lr'], weight_decay=HP['wd'])
    T = lambda arrs: [torch.as_tensor(a) for a in arrs]
    Xtr = T(inputs_tr); Xva = T(inputs_va)
    ytr = torch.as_tensor(fwd_t(y_tr), dtype=torch.float32); ltr = torch.as_tensor(l_tr, dtype=torch.float32)
    yva = torch.as_tensor(fwd_t(y_va), dtype=torch.float32); lva = torch.as_tensor(l_va, dtype=torch.float32)
    wr, wc = (1.0, lam) if lam is not None else (1.0, 1.0)
    if lam == 'reg': wr, wc = 1.0, 0.0
    if lam == 'cls': wr, wc = 0.0, 1.0
    n = len(ytr); hist = []
    best_r, best_c, st_r, st_c, bad = 1e9, 1e9, None, None, 0
    g = torch.Generator().manual_seed(seed)
    snap = lambda: {k: v.clone() for k, v in model.state_dict().items()}
    for ep in range(MAX_EP):
        model.train(); perm = torch.randperm(n, generator=g)
        for b in range(0, n, HP['batch']):
            ix = perm[b:b + HP['batch']]
            pr, pc = model(*[x[ix] for x in Xtr])
            loss = wr * F.l1_loss(pr, ytr[ix]) + wc * F.binary_cross_entropy_with_logits(pc, ltr[ix])
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pr, pc = model(*Xva)
            vr = F.l1_loss(pr, yva).item(); vmae = float(np.mean(np.abs(inv_t(pr.numpy()) - y_va)))
            vc = F.binary_cross_entropy_with_logits(pc, lva).item()
        hist.append((ep + 1, vmae, vc))
        improved = False
        # head-wise early stopping: keep the snapshot with the best validation L1 for the remaining-time head
        # and the snapshot with the best validation BCE for the lateness head (the two heads overfit at different speeds)
        if vr < best_r - 1e-5: best_r, st_r, improved = vr, snap(), True
        if vc < best_c - 1e-5: best_c, st_c, improved = vc, snap(), True
        bad = 0 if improved else bad + 1
        if bad >= HP['patience']: break
    return (make, st_r, st_c), hist

def predict_torch(m, inputs):
    make, st_r, st_c = m; net = make(); X = [torch.as_tensor(a) for a in inputs]
    with torch.no_grad():
        net.load_state_dict(st_r); net.eval(); pr, _ = net(*X)
        net.load_state_dict(st_c); net.eval(); _, pc = net(*X)
    return inv_t(pr.numpy()), torch.sigmoid(pc).numpy()

def standardize(tr, *others):
    mu = tr.mean(0); sd = tr.std(0) + 1e-6
    return [((x - mu) / sd).astype(np.float32) for x in (tr,) + others]

def seq_inputs(ix, Sside, use_num_mask=None):
    return [SA[ix], SR[ix], SN[ix], Sside[ix]]

def pma_fit_predict(tr, te):
    """Process-model average: annotated transition system (state = activity sequence + current activity, per checkpoint)."""
    def table(key):
        d = {}
        for k, rem, late in zip(tr[key], tr.rem.values, tr.late.values):
            a = d.setdefault(k, [0, 0.0, 0.0]); a[0] += 1; a[1] += rem; a[2] += late
        return d
    d1, d2, d3 = table('state'), table('state2'), table('rho')
    pr, pc = [], []
    for s, s2, r in zip(te.state, te.state2, te.rho):
        q = d1.get(s)
        if q is None or q[0] < 5: q = d2.get(s2)
        if q is None or q[0] < 5: q = d3[r]
        pr.append(q[1] / q[0]); pc.append(q[2] / q[0])
    return np.array(pr), np.array(pc)

LGB_P = dict(n_estimators=1500, learning_rate=0.03, num_leaves=31, min_child_samples=20, subsample=0.8, subsample_freq=1,
             colsample_bytree=0.8, reg_lambda=1.0, num_threads=1, verbose=-1)
def gbm_fit_predict(Xtr, ytr, ltr, Xva, yva, lva, Xte, seed):
    r = lgb.LGBMRegressor(objective='l1', random_state=seed, **LGB_P)
    r.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(50, verbose=False)])
    c = lgb.LGBMClassifier(objective='binary', random_state=seed, **LGB_P)
    c.fit(Xtr, ltr, eval_set=[(Xva, lva)], callbacks=[lgb.early_stopping(50, verbose=False)])
    return (r.predict(Xva), c.predict_proba(Xva)[:, 1]), (r.predict(Xte), c.predict_proba(Xte)[:, 1]), r, c

def best_thr(y, p):
    pr, rc, th = precision_recall_curve(y, p); f = 2 * pr * rc / (pr + rc + 1e-12)
    return th[np.argmax(f[:-1])] if len(th) else 0.5

def evaluate(te, pr, pc, thr):
    out = {}
    for rho in RHO_EVAL:
        m = (te.rho == rho).values; y = te.late.values[m]; t = te.rem.values[m]
        th = thr.get(rho, 0.5)
        out[rho] = dict(MAE=float(np.mean(np.abs(pr[m] - t))), PRAUC=float(average_precision_score(y, pc[m])),
                        ROCAUC=float(roc_auc_score(y, pc[m])), F1=float(f1_score(y, pc[m] >= th)), n=int(m.sum()), prev=float(y.mean()))
    return out

def thresholds(va, pcv):
    return {rho: best_thr(va.late.values[(va.rho == rho).values], pcv[(va.rho == rho).values]) for rho in RHO_EVAL}

# variants of the proposed TCN (ablation + sensitivity) ---------------------------------------
CTX_TIME = ['c_since_last', 'c_tod_sin', 'c_tod_cos', 'c_dow_sat', 'c_dow_sun', 'c_wip_plant', 'c_wip_cell']
TCN_VARIANTS = {
    'TCN (proposed)': {},
    'w/o run-time context': dict(drop_ctx=True),
    'w/o event attributes': dict(use_attr=False),
    'w/o in-progress token': dict(no_inprog=True),
    'w/o dilation': dict(dilate=False),
    'regression-only loss': dict(lam='reg'),
    'classification-only loss': dict(lam='cls'),
}
SENS = [(k, L) for k in (2, 3, 5) for L in (1, 2, 3)]
SENS_FOLDS = 3  # sensitivity grid on the first three folds (runtime)
if QUICK: SENS = [(3, 1), (3, 3)]

def run_tcn(tr_ix, va_ix, te_ix, seed, drop_ctx=False, use_attr=True, no_inprog=False, dilate=True, lam=1.0, k=None, levels=None,
            pool=None, side_agg=None, curve=False):
    k = k or HP['k']; levels = levels or HP['levels']; pool = pool or HP['pool']; side_agg = HP['side_agg'] if side_agg is None else side_agg
    cols = S_COLS + [c for c in C_COLS if not (drop_ctx and c in CTX_TIME)] + (AG_COLS if side_agg else [])
    side = P[cols].values.astype(np.float32)
    s_tr, s_va, s_te = standardize(side[tr_ix], side[va_ix], side[te_ix])
    A, R, N = SA.copy(), SR.copy(), SN.copy()
    if no_inprog:  # remove the in-progress token (shift sequence right, keep left padding)
        ipm = N[:, :, TOK_NUM.index('inprog')] > 0.5
        for i in np.where(ipm.any(1))[0]:
            A[i] = np.r_[0, A[i][:-1]]; R[i] = np.r_[0, R[i][:-1]]; N[i] = np.r_[np.zeros((1, N.shape[2]), np.float32), N[i][:-1]]
    mu = N[tr_ix].reshape(-1, N.shape[2]).mean(0); sd = N[tr_ix].reshape(-1, N.shape[2]).std(0) + 1e-6
    Nn = ((N - mu) / sd).astype(np.float32) * (A > 0)[:, :, None]
    make = lambda: TCN(len(ACTS), len(RESS), len(TOK_NUM), len(cols), ch=HP['ch'], k=k, levels=levels, dilate=dilate,
                       p=HP['dropout'], use_attr=use_attr, pool=pool)
    I = lambda ix, s: [A[ix], R[ix], Nn[ix], s]
    m, hist = train_torch(make, I(tr_ix, s_tr), P.rem.values[tr_ix], P.late.values[tr_ix], I(va_ix, s_va),
                          P.rem.values[va_ix], P.late.values[va_ix], seed, lam=lam)
    return predict_torch(m, I(va_ix, s_va)), predict_torch(m, I(te_ix, s_te)), hist, sum(p.numel() for p in make().parameters())

def run_mlp(tr_ix, va_ix, te_ix, seed):
    X = P[S_COLS + C_COLS + AG_COLS].values.astype(np.float32)
    xtr, xva, xte = standardize(X[tr_ix], X[va_ix], X[te_ix])
    m, hist = train_torch(lambda: MLP(X.shape[1], HP['dropout']), [xtr], P.rem.values[tr_ix], P.late.values[tr_ix], [xva],
                          P.rem.values[va_ix], P.late.values[va_ix], seed)
    return predict_torch(m, [xva]), predict_torch(m, [xte]), hist, sum(p.numel() for p in MLP(X.shape[1]).parameters())

def run_gbm(tr_ix, va_ix, te_ix, seed, cols=None):
    cols = cols or (S_COLS + C_COLS + AG_COLS)
    X = P[cols].values
    v, t, r, c = gbm_fit_predict(X[tr_ix], P.rem.values[tr_ix], P.late.values[tr_ix], X[va_ix], P.rem.values[va_ix],
                                 P.late.values[va_ix], X[te_ix], seed)
    return v, t, (r.best_iteration_, c.best_iteration_), r, c

if not FIGS_ONLY:
    # =====================================================================================
    # 4. Case-grouped 5-fold CV (validation = 15% of training cases, for early stopping + F1 threshold)
    # =====================================================================================
    rng = np.random.RandomState(SEED)
    case_ids = cases.case_id.values; perm_cases = rng.permutation(len(case_ids))
    fold_of_case = np.empty(len(case_ids), int); fold_of_case[perm_cases] = np.arange(len(case_ids)) % 5
    fold_p = fold_of_case[P.case_idx.values]
    rows, abl_rows, sens_rows, oof, curves, gbm_imp = [], [], [], {}, {}, []
    n_params = {}
    for f in range(NFOLD):
        seed = SEED + f
        te_mask = (fold_p == f) & P.is_eval.values
        trc = np.where(fold_of_case != f)[0]; vrng = np.random.RandomState(seed)
        va_cases = set(vrng.choice(trc, int(0.15 * len(trc)), replace=False))
        is_va = np.isin(P.case_idx.values, list(va_cases))
        tr_ix = np.where((fold_p != f) & ~is_va)[0]; va_ix = np.where((fold_p != f) & is_va)[0]; te_ix = np.where(te_mask)[0]
        te = P.iloc[te_ix]; va = P.iloc[va_ix]; tr = P.iloc[tr_ix]
        preds = {}
        # process-model average
        pv = pma_fit_predict(tr, va); pt = pma_fit_predict(tr, te); preds['Process-model avg.'] = (pv, pt)
        # LightGBM
        v, t, its, rmod, cmod = run_gbm(tr_ix, va_ix, te_ix, seed); preds['LightGBM'] = (v, t)
        gbm_imp.append(pd.Series(cmod.booster_.feature_importance('gain'), index=S_COLS + C_COLS + AG_COLS))
        v2, t2, _, _, _ = run_gbm(tr_ix, va_ix, te_ix, seed, cols=S_COLS + ['c_rho', 'c_tau', 'c_slack'] + AG_COLS)
        preds['LightGBM w/o run-time context'] = (v2, t2)
        # MLP
        v, t, hist, npm = run_mlp(tr_ix, va_ix, te_ix, seed); preds['MLP'] = (v, t); n_params['MLP'] = npm
        curves.setdefault('MLP', []).append(hist)
        log(f'fold {f}: baselines done')
        # TCN + ablations
        for name, kw in TCN_VARIANTS.items():
            v, t, hist, npm = run_tcn(tr_ix, va_ix, te_ix, seed, **kw); preds[name] = (v, t)
            if name == 'TCN (proposed)': n_params['TCN'] = npm; curves.setdefault('TCN (proposed)', []).append(hist)
            log(f'fold {f}: {name} done ({len(hist)} ep)')
        for name, (pv, pt) in preds.items():
            thr = thresholds(va, pv[1]); res = evaluate(te, pt[0], pt[1], thr)
            for rho, d in res.items():
                rec = dict(fold=f, model=name, rho=rho, **d)
                (rows if name in COL else abl_rows).append(rec)
                if name in TCN_VARIANTS and name != 'TCN (proposed)': pass
            if name == 'TCN (proposed)':
                for rho, d in res.items(): abl_rows.append(dict(fold=f, model=name, rho=rho, **d))
            o = oof.setdefault(name, dict(idx=[], pr=[], pc=[])); o['idx'] += list(te_ix); o['pr'] += list(pt[0]); o['pc'] += list(pt[1])
        # sensitivity of the TCN to kernel size and number of levels (receptive field)
        for (k, L) in (SENS if f < SENS_FOLDS else []):
            if (k, L) == (3, 3):
                res = evaluate(te, preds['TCN (proposed)'][1][0], preds['TCN (proposed)'][1][1], thresholds(va, preds['TCN (proposed)'][0][1]))
            else:
                v, t, _, _ = run_tcn(tr_ix, va_ix, te_ix, seed, k=k, levels=L)
                res = evaluate(te, t[0], t[1], thresholds(va, v[1]))
            for rho, d in res.items(): sens_rows.append(dict(fold=f, k=k, levels=L, rho=rho, **d))
        log(f'fold {f}: sensitivity done')
        pd.DataFrame(rows).to_csv(os.path.join(RES, 'main_per_fold.csv'), index=False)

    main = pd.DataFrame(rows); abl = pd.DataFrame(abl_rows); sens = pd.DataFrame(sens_rows)
    main.to_csv(os.path.join(RES, 'main_per_fold.csv'), index=False)
    abl.to_csv(os.path.join(RES, 'ablation_per_fold.csv'), index=False); sens.to_csv(os.path.join(RES, 'sensitivity_per_fold.csv'), index=False)

    def ms(df, keys, metrics=('MAE', 'PRAUC', 'F1', 'ROCAUC')):
        g = df.groupby(keys)
        out = g[list(metrics)].agg(['mean', 'std'])
        out.columns = [f'{a}_{b}' for a, b in out.columns]
        return out.reset_index()
    main_t = ms(main, ['rho', 'model']); main_t.to_csv(os.path.join(RES, 'table_main_comparison.csv'), index=False)
    abl_t = ms(abl, ['model', 'rho']); abl_t.to_csv(os.path.join(RES, 'table_ablation.csv'), index=False)
    sens_t = ms(sens, ['k', 'levels', 'rho']); sens_t.to_csv(os.path.join(RES, 'table_sensitivity.csv'), index=False)
    # averaged over checkpoints
    abl_avg = abl.groupby(['model', 'fold'])[['MAE', 'PRAUC', 'F1']].mean().groupby('model').agg(['mean', 'std'])
    abl_avg.columns = [f'{a}_{b}' for a, b in abl_avg.columns]; abl_avg.reset_index().to_csv(os.path.join(RES, 'table_ablation_avg.csv'), index=False)
    log('CV done')

    # =====================================================================================
    # 5. Significance: pooled out-of-fold paired tests (TCN vs each baseline)
    # =====================================================================================
    sig = []
    O = {k: pd.DataFrame(dict(idx=v['idx'], pr=v['pr'], pc=v['pc'])).set_index('idx').sort_index() for k, v in oof.items()}
    brng = np.random.RandomState(SEED)
    for rho in RHO_EVAL:
        ix = O['TCN (proposed)'].index[P.rho.values[O['TCN (proposed)'].index] == rho]
        y = P.late.values[ix]; t = P.rem.values[ix]
        et = np.abs(O['TCN (proposed)'].loc[ix, 'pr'].values - t); pt = O['TCN (proposed)'].loc[ix, 'pc'].values
        for b in ['LightGBM', 'MLP', 'Process-model avg.']:
            eb = np.abs(O[b].loc[ix, 'pr'].values - t); pb = O[b].loc[ix, 'pc'].values
            p_mae = wilcoxon(et, eb).pvalue
            boots = []
            for _ in range(1000):
                s = brng.randint(0, len(ix), len(ix))
                if y[s].sum() == 0: continue
                boots.append(average_precision_score(y[s], pt[s]) - average_precision_score(y[s], pb[s]))
            boots = np.array(boots)
            sig.append(dict(rho=rho, baseline=b, dMAE=float(et.mean() - eb.mean()), p_wilcoxon=float(p_mae),
                            dPRAUC=float(average_precision_score(y, pt) - average_precision_score(y, pb)),
                            dPRAUC_lo=float(np.percentile(boots, 2.5)), dPRAUC_hi=float(np.percentile(boots, 97.5))))
    sig = pd.DataFrame(sig); sig.to_csv(os.path.join(RES, 'significance.csv'), index=False)
    log('significance done')

    # =====================================================================================
    # 6. Temporal hold-out (train on first 80% of releases, test on the last 20%), 5 seeds for neural models
    # =====================================================================================
    cut = np.quantile(cases.rel_h.values, 0.8)
    late_case = cases.rel_h.values[P.case_idx.values] > cut
    tmp_rows = []
    tr_all = np.where(~late_case)[0]
    early_cases = np.unique(P.case_idx.values[tr_all])
    vcut = np.quantile(cases.rel_h.values[early_cases], 0.85)
    is_va = cases.rel_h.values[P.case_idx.values] > vcut
    tr_ix = np.where(~late_case & ~is_va)[0]; va_ix = np.where(~late_case & is_va)[0]; te_ix = np.where(late_case & P.is_eval.values)[0]
    te = P.iloc[te_ix]; va = P.iloc[va_ix]
    nseed = 2 if QUICK else 5
    for s in range(nseed):
        seed = SEED + 100 + s
        out = {}
        out['TCN (proposed)'] = run_tcn(tr_ix, va_ix, te_ix, seed)[:2]
        out['MLP'] = run_mlp(tr_ix, va_ix, te_ix, seed)[:2]
        out['LightGBM'] = run_gbm(tr_ix, va_ix, te_ix, seed)[:2]
        if s == 0: out['Process-model avg.'] = (pma_fit_predict(P.iloc[tr_ix], va), pma_fit_predict(P.iloc[tr_ix], te))
        for name, (pv, pt) in out.items():
            res = evaluate(te, pt[0], pt[1], thresholds(va, pv[1]))
            for rho, d in res.items(): tmp_rows.append(dict(seed=s, model=name, rho=rho, **d))
        log(f'temporal seed {s} done')
    tmp = pd.DataFrame(tmp_rows); tmp.to_csv(os.path.join(RES, 'temporal_holdout_per_seed.csv'), index=False)
    tmp_avg = tmp.groupby(['model', 'seed'])[['MAE', 'PRAUC', 'F1']].mean().groupby('model').agg(['mean', 'std'])
    tmp_avg.columns = [f'{a}_{b}' for a, b in tmp_avg.columns]; tmp_avg = tmp_avg.reset_index()
    tmp_avg.to_csv(os.path.join(RES, 'table_temporal_holdout.csv'), index=False)

    # =====================================================================================
    # 7. Bottleneck mining from the log (waiting-time attribution, CART hotspots, what-if bound)
    # =====================================================================================
    E = ev.copy()
    E['wait_h'] = E.waiting_min / 60
    ready = E.start_timestamp - pd.to_timedelta(E.waiting_min, unit='m')
    E['ready_hour'] = ready.dt.hour; E['ready_dow'] = ready.dt.dayofweek
    E = E.merge(cases[['case_id', 'product_family', 'complexity_index', 'order_quantity', 'priority', 'material_shortage', 'late_completion',
                       'cycle_time_hours', 'due_in_hours']], on='case_id', how='left')
    E['loop_idx'] = E.groupby('case_id').activity.transform(lambda s: (s == 'rework').cumsum())
    E['prev_fault'] = E.groupby('case_id').robot_fault.shift().fillna(0)
    E['weld_fault_before'] = E.groupby('case_id').apply(lambda g: ((g.activity == 'robotic_welding') & (g.robot_fault == 1)).cumsum().shift().fillna(0)).reset_index(level=0, drop=True).clip(upper=1)
    tot_wait = E.wait_h.sum()
    attr = E.groupby(['activity', 'resource']).agg(events=('wait_h', 'size'), total_wait_h=('wait_h', 'sum'), mean_wait_h=('wait_h', 'mean'),
                                                  median_wait_h=('wait_h', 'median')).reset_index()
    lw = E.groupby(['activity', 'resource', 'late_completion']).wait_h.mean().unstack()
    attr['mean_wait_late_h'] = [lw.loc[(a, r)].get(1, np.nan) for a, r in zip(attr.activity, attr.resource)]
    attr['mean_wait_ontime_h'] = [lw.loc[(a, r)].get(0, np.nan) for a, r in zip(attr.activity, attr.resource)]
    attr['share_pct'] = 100 * attr.total_wait_h / tot_wait
    attr = attr.sort_values('total_wait_h', ascending=False); attr.to_csv(os.path.join(RES, 'table_waiting_attribution.csv'), index=False)
    # activity-level share
    act_share = E.groupby('activity').wait_h.sum().sort_values(ascending=False) / tot_wait * 100
    # CART hotspot mining on event waiting time (descriptive, uses attributes of each event)
    FEATS = pd.get_dummies(E[['activity', 'resource', 'product_family', 'production_cell', 'priority']].astype(str), prefix_sep='=')
    FEATS['complexity'] = E.complexity_index; FEATS['qty'] = E.order_quantity; FEATS['shortage'] = E.material_shortage
    FEATS['ready_hour'] = E.ready_hour; FEATS['ready_dow'] = E.ready_dow; FEATS['loop_idx'] = E.loop_idx
    FEATS['prev_fault'] = E.prev_fault; FEATS['weld_fault_before'] = E.weld_fault_before
    tree = DecisionTreeRegressor(max_depth=HOT_DEPTH, min_samples_leaf=50, random_state=SEED).fit(FEATS.values, E.wait_h.values)
    leaf = tree.apply(FEATS.values)
    def leaf_rule(tree, names):
        t = tree.tree_; rules = {}
        def rec(n, conds):
            if t.children_left[n] == -1: rules[n] = conds; return
            nm, th = names[t.feature[n]], t.threshold[n]
            if '=' in nm: l, r = f'{nm.split("=")[0]}!={nm.split("=")[1]}', nm
            else: l, r = f'{nm}<={th:.2f}', f'{nm}>{th:.2f}'
            rec(t.children_left[n], conds + [l]); rec(t.children_right[n], conds + [r])
        rec(0, []); return rules
    rules = leaf_rule(tree, list(FEATS.columns))
    DOW = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    def pretty(v, op, th):
        if v == 'ready_dow':
            lo, hi = (0, int(np.floor(th))) if op == '<=' else (int(np.floor(th)) + 1, 6)
            return 'ready ' + (DOW[lo] if lo == hi else f'{DOW[lo]}-{DOW[hi]}')
        if v == 'ready_hour': return f'ready {"<" if op == "<=" else ">="}{int(np.floor(th)) + 1:02d}:00'
        if v == 'loop_idx': return f'loop {"<=" if op == "<=" else ">="}{int(np.floor(th)) + (0 if op == "<=" else 1)}'
        if v in ('weld_fault_before', 'prev_fault', 'shortage'):
            nm = {'weld_fault_before': 'after weld fault', 'prev_fault': 'after fault', 'shortage': 'shortage'}[v]
            return nm if op == '>' else f'no {nm}'
        return f'{v}{op}{th:.2f}'
    def simplify(conds):  # positive category conditions + tightest numeric bounds, in readable form
        pos = [c.split('=', 1)[1] for c in conds if '=' in c and '!=' not in c and '<=' not in c]
        num = {}
        for c in conds:
            for op in ('<=', '>'):
                if op in c:
                    v, th = c.split(op); th = float(th); key = (v, op)
                    num[key] = min(num.get(key, th), th) if op == '<=' else max(num.get(key, th), th)
                    break
        # merge two-sided bounds on the same variable
        out = []
        for (v, op), th in num.items():
            if op == '>' and (v, '<=') in num and v in ('ready_dow', 'ready_hour'):
                lo, hi = int(np.floor(th)) + 1, int(np.floor(num[(v, '<=')]))
                out.append('ready ' + (f'{DOW[lo]}' if lo == hi else f'{DOW[lo]}-{DOW[hi]}') if v == 'ready_dow' else f'ready {lo:02d}-{hi + 1:02d}h')
            elif op == '<=' and (v, '>') in num and v in ('ready_dow', 'ready_hour'):
                continue
            else:
                out.append(pretty(v, op, th))
        return ', '.join(pos + out) or 'all other events'
    hot = pd.DataFrame(dict(leaf=leaf, wait_h=E.wait_h.values, act=E.activity.values)).groupby('leaf').agg(
        events=('wait_h', 'size'), mean_wait_h=('wait_h', 'mean'), total_wait_h=('wait_h', 'sum'),
        act=('act', lambda s: s.mode().iloc[0])).reset_index()
    hot['rule'] = [simplify(rules[l]) for l in hot.leaf]; hot['share_pct'] = 100 * hot.total_wait_h / tot_wait
    # log-based what-if bound: scale the waiting of every event in a hotspot so that its mean equals the mean
    # waiting of the same activity outside the hotspot; subtract the saving from the case's cycle time
    # (first-order: ignores queue interactions / knock-on effects on other cases)
    case_cycle = cases.set_index('case_id'); whatif = []
    for _, h in hot.iterrows():
        m = leaf == h.leaf; sub = E[m]
        save = pd.Series(0.0, index=sub.index)
        for a_, g_ in sub.groupby('activity'):
            ref = E.wait_h[(~m) & (E.activity == a_)].mean(); hm = g_.wait_h.mean()
            if hm > ref > 0: save[g_.index] = g_.wait_h * (1 - ref / hm)
        per_case = save.groupby(sub.case_id).sum(); per_case = per_case[per_case > 0]
        cc = case_cycle.loc[per_case.index]
        newly = int(((cc.cycle_time_hours > cc.due_in_hours) & (cc.cycle_time_hours - per_case <= cc.due_in_hours)).sum())
        late_touch = int((cc.cycle_time_hours > cc.due_in_hours).sum())
        whatif.append(dict(leaf=h.leaf, saved_h_total=float(save.sum()), cases_affected=int(len(per_case)), late_cases_touched=late_touch,
                           mean_saved_per_case_h=float(per_case.mean()) if len(per_case) else 0.0, late_cases_recovered=newly))
    hot = hot.merge(pd.DataFrame(whatif), on='leaf').sort_values('saved_h_total', ascending=False)
    hot.to_csv(os.path.join(RES, 'table_bottleneck_hotspots.csv'), index=False)
    n_late = int(cases.late_completion.sum())
    # resource x ready-hour heat map of mean waiting
    res_top = attr.groupby('resource').total_wait_h.sum().sort_values(ascending=False).index[:10]
    H = E[E.resource.isin(res_top)].assign(hb=lambda d: d.ready_hour // 2 * 2).pivot_table(index='resource', columns='hb', values='wait_h', aggfunc='mean')
    H = H.loc[res_top]; H.to_csv(os.path.join(RES, 'table_wait_resource_hour.csv'))
    Hn = E[E.resource.isin(res_top)].assign(hb=lambda d: d.ready_hour // 2 * 2).pivot_table(index='resource', columns='hb', values='wait_h', aggfunc='size').loc[res_top]

    log('bottleneck mining done')

    # =====================================================================================
    # 8. Dataset summary table
    # =====================================================================================
    ds = []
    ds.append(('Work orders (cases) / events', f'{len(cases):,} / {len(ev):,}'))
    ds.append(('Release period', f'{cases.release_timestamp.min():%Y-%m-%d} to {cases.release_timestamp.max():%Y-%m-%d}'))
    ds.append(('Activities / resources', f'{len(ACTS)} / {len([r for r in RESS if r != "UNK"])}'))
    ds.append(('Events per case (min / median / max)', f'{cases.n_events.min()} / {int(cases.n_events.median())} / {cases.n_events.max()}'))
    ds.append(('Cycle time h (median / P90)', f'{cases.cycle_time_hours.median():.1f} / {cases.cycle_time_hours.quantile(.9):.1f}'))
    ds.append(('Late completions', f'{n_late} ({100*cases.late_completion.mean():.1f}%)'))
    ds.append(('Cases with rework / robot fault', f'{100*(cases.n_rework_loops>0).mean():.1f}% / {100*cases.any_robot_fault.mean():.1f}%'))
    for rho in RHO_EVAL:
        m = P.rho == rho
        ds.append((f'Running prefixes at rho={rho} (late share)', f'{int(m.sum())} ({100*P.late[m].mean():.1f}%)'))
    ds.append(('Missing: resource / load / complexity / qty', f'{prep["resource_imputed"]} / {prep["plant_load_imputed"]} / {prep["complexity_index_imputed"]} / {prep["order_quantity_imputed"]}'))
    ds.append(('Repaired outliers: duration / waiting', f'{prep["duration_outliers_repaired"]} / {prep["waiting_outliers_repaired"]}'))
    pd.DataFrame(ds, columns=['Item', 'Value']).to_csv(os.path.join(RES, 'table_dataset_summary.csv'), index=False)


    CK_NAMES = ['main', 'abl', 'sens', 'main_t', 'abl_t', 'sens_t', 'abl_avg', 'O', 'curves', 'gbm_imp', 'n_params', 'sig', 'tmp_avg',
                'hot', 'H', 'Hn', 'attr', 'act_share', 'tot_wait', 'n_late']
    pickle.dump({k: globals()[k] for k in CK_NAMES}, open(os.path.join(HERE, '..', 'results', 'raw_checkpoint.pkl'), 'wb'))

else:  # redraw figures from the saved checkpoint of a previous full run
    globals().update(pickle.load(open(os.path.join(RES, 'raw_checkpoint.pkl'), 'rb')))

# =====================================================================================
# 9. Figures
# =====================================================================================
MODELS = ['TCN (proposed)', 'LightGBM', 'MLP', 'Process-model avg.']
SHORT = {'TCN (proposed)': 'TCN', 'LightGBM': 'LGBM', 'MLP': 'MLP', 'Process-model avg.': 'PMA'}
FW = 3.0   # figures are drawn at their printed column width (in) so that every font prints at >= 7 pt
# Fig 2: MAE and PR-AUC vs checkpoint
fig, ax = plt.subplots(1, 2, figsize=(FW, 1.75))
xs = np.arange(3); w = 0.2
for i, mdl in enumerate(MODELS):
    d = main_t[main_t.model == mdl].set_index('rho').loc[RHO_EVAL]
    lab = 'TCN (proposed)' if mdl.startswith('TCN') else SHORT[mdl]
    ax[0].bar(xs + (i - 1.5) * w, d.MAE_mean, w, yerr=d.MAE_std, color=COL[mdl], label=lab, capsize=1.2, error_kw=dict(lw=0.6))
    ax[1].bar(xs + (i - 1.5) * w, d.PRAUC_mean, w, yerr=d.PRAUC_std, color=COL[mdl], capsize=1.2, error_kw=dict(lw=0.6))
prev = [P.late[(P.rho == r)].mean() for r in RHO_EVAL]
ax[1].plot(xs, prev, 'k:', marker='_', lw=0.8, label='late base rate')
for a, yl in zip(ax, ['Remaining-time MAE (h)', 'Late-completion PR-AUC']):
    a.set_xticks(xs); a.set_xticklabels([f'{int(r*100)}%' for r in RHO_EVAL]); a.set_ylabel(yl); a.set_xlabel(r'Checkpoint $\rho$')
    a.spines[['top', 'right']].set_visible(False)
ax[1].set_ylim(0, 1)
h1, l1 = ax[0].get_legend_handles_labels(); h2, l2 = ax[1].get_legend_handles_labels()
fig.legend(h1 + h2, l1 + l2, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.2), frameon=False, fontsize=7, columnspacing=0.8, handlelength=1.2)
fig.tight_layout(w_pad=0.4); savefig(fig, 'fig2_main_comparison')

# Fig 3: pooled out-of-fold PR curves at rho = 0.5 + predicted vs actual remaining time
fig, ax = plt.subplots(1, 2, figsize=(FW, 1.8))
rho = 0.5
for mdl in MODELS:
    o = O[mdl]; ix = o.index[P.rho.values[o.index] == rho]
    pr_, rc_, _ = precision_recall_curve(P.late.values[ix], o.loc[ix, 'pc'])
    ax[0].plot(rc_, pr_, color=COL[mdl], lw=1.0, label=f'{SHORT[mdl]} {average_precision_score(P.late.values[ix], o.loc[ix, "pc"]):.2f}')
ax[0].axhline(P.late.values[ix].mean(), color='k', ls=':', lw=0.7)
ax[0].set_xlabel('Recall'); ax[0].set_ylabel('Precision'); ax[0].set_ylim(0.3, 1.02)
fig.legend(*ax[0].get_legend_handles_labels(), loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.17), frameon=False, fontsize=7,
           columnspacing=0.7, handlelength=1.0, title=r'pooled out-of-fold PR-AUC at $\rho$ = 50%', title_fontsize=7)
o = O['TCN (proposed)']; ix = o.index[P.rho.values[o.index] == rho]
ax[1].scatter(P.rem.values[ix], o.loc[ix, 'pr'], s=1.5, alpha=0.35, color=COL['TCN (proposed)'], lw=0)
mx = np.percentile(P.rem.values[ix], 99); ax[1].plot([0, mx], [0, mx], 'k--', lw=0.6)
ax[1].set_xlim(0, mx); ax[1].set_ylim(0, mx); ax[1].set_xlabel('Actual remaining (h)'); ax[1].set_ylabel('TCN prediction (h)')
for a in ax: a.spines[['top', 'right']].set_visible(False)
fig.tight_layout(w_pad=0.6); savefig(fig, 'fig3_pr_scatter')

# Fig 6: bottleneck hotspots (top 8 leaves by what-if saving)
top = hot.head(8).iloc[::-1]
def short(r):
    for a, b in [('robotic_welding', 'Welding'), ('robotic_assembly', 'Assembly'), ('quality_inspection', 'QI'), ('QI-HUMAN-1', 'Hum. QI'),
                 ('no after weld fault', 'no weld fault'), (', loop <=0', ''), ('loop >=1', 'rework'), ('ready ', ''),
                 ('complexity', 'cplx'), ('normal', 'normal prio.'), ('>=', '≥'), ('<=', '≤'), ('Sat-Sun', 'wkend'), ('Mon-Fri', 'wkday'),
                 ('Sun', 'Sun'), (':00', 'h')]:
        r = r.replace(a, b)
    return r
fig, ax = plt.subplots(figsize=(FW, 2.25))
yy = np.arange(len(top))
ax.barh(yy, top.total_wait_h / 1000, color='#bdbdbd', label='total waiting')
ax.barh(yy, top.saved_h_total / 1000, color='#c0392b', height=0.45, label=r'what-if bound $\Delta W_h$')
for y_, (_, r) in zip(yy, top.iterrows()):
    ax.text(r.total_wait_h / 1000 + 0.15, y_, f'{r.mean_wait_h:.0f} h, +{r.late_cases_recovered}', va='center', fontsize=7)
ax.set_yticks(yy); ax.set_yticklabels([short(r) for r in top.rule], fontsize=7)
ax.set_xlabel('Waiting time (thousand h)')
ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.2, 1.15), ncol=2, fontsize=7, columnspacing=0.8, handlelength=1.2)
ax.set_xlim(0, top.total_wait_h.max() / 1000 * 1.6); ax.spines[['top', 'right']].set_visible(False)
fig.tight_layout(); savefig(fig, 'fig6_hotspots')

# Fig 5: waiting heat map resource x ready hour (cells with < 15 events masked)
Hm = H.where(Hn >= 15)
fig, ax = plt.subplots(figsize=(FW, 1.85))
cmap = plt.get_cmap('Reds').copy(); cmap.set_bad('#e8e8e8')
im = ax.imshow(np.ma.masked_invalid(Hm.values), aspect='auto', cmap=cmap)
ax.set_xticks(range(0, H.shape[1], 2)); ax.set_xticklabels([f'{int(c):02d}' for c in H.columns[::2]], fontsize=7)
ax.set_yticks(range(H.shape[0])); ax.set_yticklabels(H.index, fontsize=7)
ax.set_xlabel('Hour when the order became ready')
cb = fig.colorbar(im, ax=ax, pad=0.02); cb.set_label('Mean waiting (h)', fontsize=7); cb.ax.tick_params(labelsize=7)
fig.tight_layout(); savefig(fig, 'fig5_wait_heatmap')

# Fig 4: sensitivity heat map (k x levels), averaged over checkpoints and folds 0-2, + learning curves
sv = sens.groupby(['k', 'levels', 'fold'])[['MAE', 'PRAUC']].mean().groupby(['k', 'levels']).mean().reset_index()
fig, ax = plt.subplots(1, 2, figsize=(FW, 1.7))
Ks = sorted(sv.k.unique()); Ls = sorted(sv.levels.unique())
M = sv.pivot(index='k', columns='levels', values='MAE').loc[Ks, Ls]
Mp = sv.pivot(index='k', columns='levels', values='PRAUC').loc[Ks, Ls]
im = ax[0].imshow(M.values, cmap='Reds_r', aspect='auto', vmin=M.values.min() - 0.3, vmax=M.values.max() + 0.3)
for i in range(len(Ks)):
    for j in range(len(Ls)): ax[0].text(j, i, f'{M.values[i, j]:.2f}\n{Mp.values[i, j]:.3f}', ha='center', va='center', fontsize=7)
ax[0].set_xticks(range(len(Ls))); ax[0].set_xticklabels(Ls); ax[0].set_yticks(range(len(Ks))); ax[0].set_yticklabels(Ks)
ax[0].set_xlabel('Residual levels $L$'); ax[0].set_ylabel('Kernel size $k$'); ax[0].set_title('MAE (h) / PR-AUC', fontsize=7)
for mdl in ['TCN (proposed)', 'MLP']:
    hs = curves[mdl]; ml = min(len(h) for h in hs)  # epochs reached by every fold
    arr = np.array([[x[1] for x in h[:ml]] for h in hs])
    ax[1].plot(np.arange(1, ml + 1), arr.mean(0), color=COL[mdl], lw=1, label=SHORT[mdl])
    ax[1].fill_between(np.arange(1, ml + 1), arr.min(0), arr.max(0), color=COL[mdl], alpha=0.15, lw=0)
ax[1].set_xlabel('Epoch'); ax[1].set_ylabel('Validation MAE (h)'); ax[1].legend(frameon=False, fontsize=7)
ax[1].spines[['top', 'right']].set_visible(False)
fig.tight_layout(w_pad=0.6); savefig(fig, 'fig4_sensitivity_curve')

# =====================================================================================
# 10. metrics.json
# =====================================================================================
imp = pd.concat(gbm_imp, axis=1).mean(1).sort_values(ascending=False)
avg = main.groupby(['model', 'fold'])[['MAE', 'PRAUC', 'F1', 'ROCAUC']].mean().groupby('model').agg(['mean', 'std'])
metrics = dict(
    preprocessing=prep, n_prefixes_train_grid=int(len(P)), n_eval_prefixes=int(P.is_eval.sum()), n_params=n_params,
    main={m: {str(r): main_t[(main_t.model == m) & (main_t.rho == r)].iloc[0].drop(['model', 'rho']).to_dict() for r in RHO_EVAL} for m in MODELS},
    main_avg_over_checkpoints={m: {f'{a}_{b}': float(avg.loc[m, (a, b)]) for a in ['MAE', 'PRAUC', 'F1', 'ROCAUC'] for b in ['mean', 'std']} for m in MODELS},
    ablation_avg=abl_avg.to_dict(orient='index'), temporal_holdout=tmp_avg.set_index('model').to_dict(orient='index'),
    significance=sig.to_dict(orient='records'), gbm_top_features_gain=imp.head(12).to_dict(),
    bottleneck=dict(total_wait_h=float(tot_wait), activity_share_pct=act_share.round(2).to_dict(), n_late=n_late,
                    hotspots=hot.head(10).to_dict(orient='records')),
    sensitivity=sv.to_dict(orient='records'), runtime_s=time.time() - T0, quick=QUICK)
if not FIGS_ONLY: json.dump(metrics, open(os.path.join(RES, "metrics.json"), "w"), indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
log('ALL DONE, runtime %.1f s' % (time.time() - T0))

# Fig 1: methodology flowchart (Graphviz)
import subprocess
from PIL import Image
subprocess.run(['dot', '-Tpng', '-Gdpi=300', os.path.join(HERE, 'fig1_flowchart.dot'), '-o', os.path.join(FIG, 'fig1_flowchart.png')], check=True)
Image.open(os.path.join(FIG, 'fig1_flowchart.png')).convert('RGB').save(os.path.join(RES, 'fig1_flowchart.jpg'), quality=92)
log('flowchart rendered')
