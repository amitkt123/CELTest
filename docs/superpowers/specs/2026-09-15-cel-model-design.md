# CEL: A Compute–Energy–Labor Model of Machine Intelligence — Design Spec

**Date:** 2026-09-15
**Status:** Approved section by section; reconstructed from `cel/` (build steps 1–2) and extended.
**Scope of this document:** the full model (build steps 0–10). Sections marked *Recovered* describe behavior that exists in code today; sections marked *New* are design to be built.

---

## §0 Conventions

| Symbol | Meaning | Unit |
|---|---|---|
| t | period (year), t0 = 2023, horizon T = 37 → 2060 | yr |
| τ | t − t0 | yr |
| Y_s | seconds per year = 3.156e7 | s |
| h_Y | hours per year = 8766 | h |
| v | hardware vintage (install year) | yr |
| i | task index on the Sobol grid | — |

Money is constant 2024 USD. Compute stocks are FLOP/s; compute flows FLOP. Power kW, energy kWh, emissions tCO₂. The final good is the numeraire, P_Y = 1. **One unit of task output is one worker-year of output at human productivity γ = 1.** L̄ = 1 corresponds to the global labor force (≈3.5e9 workers).

Every scenario is a diff against the frozen baseline `S0` (`Params.with_(**diff)`), so no scenario can drift from the baseline silently.

---

## §1 Thesis

**Working title:** *The Commodification of Cognition: A Compute–Energy–Labor Model of the Returns to, and Incidence of, Machine Intelligence.*

**Central thesis.** Machine intelligence commodifies cognitive labor: its price converges toward the marginal cost of compute and energy. This produces three linked consequences:

1. its developers capture little of the value they create;
2. the extent of automation is ultimately bounded by power and capital depreciation, not only by capability;
3. the dominant human cost arises from the *speed* of the transition, not from a permanent end to human work.

The claims are coupled: what makes AI cheap enough to displace labor rapidly is also what makes it hard for its builders to profit.

### Hypotheses

| | Statement | Falsified if | Tested in |
|---|---|---|---|
| **H1** Rent dissipation | With several near-symmetric competitors, 2–4-year hardware obsolescence, and fast algorithmic diffusion, the private NPV of frontier model development is negative while net social value is positive. Rents flow to consumers and to bottleneck owners (chips, power, land). | NPV_lab > 0 across plausible competition and depreciation settings | §2, §6, §8 |
| **H2** Physical bound | Effective automation is bounded by min(capability, compute the stock can serve); compute is bounded by power buildout and depreciation-driven scrapping. For a significant share of tasks through 2040 the binding constraint is power, not capability. | The power constraint never binds before capability saturates | §2–§4 |
| **H3** Requirement ≠ employment | The labor requirement ℓ, the labor share s_L, the wage w, and employment E are distinct. "1/50 of humans needed" is a statement about ℓ; whether it maps into low wages or joblessness depends on σ and ν. | s_L and E track ℓ one-for-one across σ | §5 |
| **H4** Transition dominance | Welfare losses concentrate in the adjustment period, governed by displacement speed relative to absorption and retraining capacity. When displacement outruns absorption, durations and scarring grow nonlinearly. | Transition losses are small relative to long-run gains even at the fastest plausible automation speed | §7, §8 |
| **H5** Self-limiting displacement | Unprofitability tightens financing → slower buildout → slower automation. "Unprofitable" and "near-total rapid displacement" coexist only under specific conditions (hyperscaler cross-subsidy, state support, cheap power), which the model identifies. | Both occur together without any of those supports | §6, §11 |

**Stylized facts** (AI capex, lab revenues, AI power use) are data to be sourced and graded in §10. No premise of the model rests on press figures.

**Deliverables:** a verdict on each hypothesis with sensitivity bounds (§8.5), and the parameter region where the strong claim (ℓ ≤ 0.02 **and** NPV_lab < 0) holds, with its plausibility.

**Scope:** global, 2023–2060, one representative economy. A US/RoW split is deferred.

---

## §2 Hardware capital and depreciation (New; parameters exist in `params.py`)

**Input:** investment I(t) in $ from §6. **Outputs:** K, C_inf, C_train, P_draw, r_cost, δ_econ, R_H, V(v,t), T*(v).

### 2.1 Frontier technology

Efficiency (FLOP/s per kW, all-in) follows a logistic toward the physical ceiling ε_max = H·ε₀ (H = `eps_headroom`):

```
ε(t) = ε_max / (1 + (H − 1)·e^{−g·τ}),        g = g_eps / (1 − 1/H)
```

The choice of g makes the initial growth rate exactly `g_eps`.

Quality-adjusted price per FLOP/s falls with efficiency and with residual fab learning:

```
κ(t) = κ₀ · (ε₀/ε(t)) · e^{−g_fab·τ}
```

The implied initial decline is g_eps + g_fab ≈ 0.31/yr. `g_kappa0` is therefore **not a free parameter**; it becomes a calibration check (must come out ≈ 0.30). κ includes the hardware vendor markup m_H; vendor rent is R_H(t) = (1 − 1/m_H)·I(t).

### 2.2 Vintages and stocks

```
K(t)     = Σ_v (I(v)/κ(v)) · e^{−δ_phys·(t−v)} · 1[v not scrapped at t]      FLOP/s installed
C_inf(t) = util·(1 − s_T)·K(t)                                                  FLOP/s delivered
C_train  = util·s_T·K(t)
P_draw   = PUE · Σ_v K_v / ε(v)                                                 kW installed
```

### 2.3 Long-run rental price

Compute rental is competitive. The long-run price that governs investment is the user cost of the frontier vintage (Hall–Jorgenson), with capital loss from expected price decline ĝ_κ and energy paid only on utilized hours:

```
r_cost(t) = κ(t)·(ρ + δ_phys + ĝ_κ) + util·h_Y·PUE·p_E(t)/ε(t)        $ per (FLOP/s)·yr
```

Markups on AI *services* are applied at the model-provider layer (§6), never here. The within-period market price r(t) clears on the vintage merit order (§9.2) and may differ from r_cost.

### 2.4 Scrapping and Proposition 1

A vintage is scrapped when its quasi-rent falls below its own operating (energy) cost. When power binds (§3), it is scrapped earlier: when its quasi-rent per kW falls below the shadow price of power λ_P. Power scarcity therefore shortens the economic life of inefficient vintages.

**Proposition 1 (economic life).** With constant growth rates and slack power, let s_E be the energy share of the frontier user cost at installation, g_r the rate at which r falls, and π_E the drift in electricity prices. Then

```
T*     = −ln(s_E) / (g_r + π_E)
δ_econ = δ_phys + g_r + (scrapping hazard)
V(v,t) = PV of quasi-rents of vintage v over [t, v + T*]
```

**Limiting case (test, step 4):** with p_E = 0, T* = ∞ and V falls at exactly δ_phys + g_κ (Hall 1968).

**H1 output:** the *depreciation gap* = straight-line book value over a 5–6-year accounting life minus economic value V. Reported profits are overstated whenever T* is shorter than the book life.

### 2.5 Unit consistency

The accelerator unit is 1 kW all-in, and κ₀·ε₀·unit_kW is the $ price of one unit. Build step 0 corrects the current inconsistencies (Appendix B).

---

## §3 Power and energy (New; parameters exist)

**Input:** P_draw from §2. **Outputs:** P̄, binding flag, λ_P, R_P, p_E, E, Em, social energy cost.

### 3.1 Power ceiling

P̄(t) is grid plus on-site capacity available to AI:

```
g_P(t)  = g_P∞ + (g_P0 − g_P∞)·e^{−τ/τ_P}
ln P̄(t) = ln P̄₀ + g_P∞·τ + (g_P0 − g_P∞)·τ_P·(1 − e^{−τ/τ_P})
```

At S0 values P̄ rises from 15 GW (2023) to ≈45 GW (2026). P̄ is exogenous; a larger P̄ is a scenario lever (S2, S4, S6), not an equilibrium outcome. This keeps H2 cleanly testable.

### 3.2 Constraint and scarcity rent

P_draw(t) ≤ P̄(t). When it binds, the within-period price r exceeds r_cost (§9.2). The premium is a rent per kW accruing to owners of grid access, sites, and generation:

```
λ_P = (r − r_cost)·ε(t)/PUE ≥ 0         only when the power constraint binds
R_P = λ_P·P̄
```

If instead finance or the investment lag binds, the same premium is a quasi-rent to compute owners, not to power owners.

### 3.3 Electricity price

Prices drift, faster in years when power binds; binding years ratchet the price level:

```
d ln p_E/dt = π_E + η_E·1[P_draw = P̄]
```

### 3.4 Energy, emissions, social cost

```
E(t)   = h_Y · util · PUE · Σ_v K_v/ε(v)                         kWh
ι(t)   = ι₀·e^{−g_ι·τ}
Em(t)  = ι(t)·E(t) + ι_emb·I(t)                                   tCO₂
SocialCost_E(t) = p_E·E + SCC·Em − τ_C·Em
```

ι_emb is embodied manufacturing emissions per $ of hardware; fast obsolescence multiplies it. τ_C is a carbon price (0 in S0). SCC is constant in real terms.

### 3.5 Energy per displaced worker-equivalent (H2 output)

Inference energy divided by worker-equivalents displaced, reported alongside human metabolic references: ≈175 kWh/yr (brain, 20 W) and ≈880 kWh/yr (whole body, 100 W).

**Excluded (limitations):** water use; spillovers of AI demand onto household and industrial electricity prices; endogenous power supply response.

---

## §4 Capability (4.3, 4.4 Recovered from `grid.py`; remainder New)

**Input:** C_train from §2. **Outputs:** x, Ω, Ω_inf, φ_cap, H, capable_i, e(i), c_AI(i).

### 4.1 Algorithmic progress

Training and inference multipliers both grow at decaying rates and share the exhaustion timescale τ_Ω:

```
ln Ω(t)     = g_Ω0  · τ_Ω · (1 − e^{−τ/τ_Ω})          S0 cumulative ≈ 120×
ln Ω_inf(t) = g_Ωinf· τ_Ω · (1 − e^{−τ/τ_Ω})          S0 cumulative ≈ 600×
```

(The current code treats Ω_inf as growing at 0.80/yr indefinitely, ≈10¹³× by 2060; this is replaced.)

### 4.2 Frontier effective compute

```
x(t) = log10( ς · C_train(t) · Θ · Y_s · Ω(t) )
```

ς is the share of all AI training compute that goes into the single largest run. It is a **class-D calibration identity** pinned by the target x(t0) = 26 (`x_t0` becomes a target, not a free input). At current S0 values (P̄₀ = 15 GW, ε₀ = 1.4e15), putting all training compute into one run would give x = 28.3, which implies ς ≈ 0.005. Either ς is that small or P̄₀/ε₀ are overstated; §10 resolves it.

### 4.3 Coverage

Each task carries an automatability flag independent of its difficulty, drawn from a fourth Sobol dimension:

```
a_i ~ Bernoulli(φ_max),  independent of d_i
capable_i(t) = [d_i ≤ x(t)] ∧ [a_i = 1]
φ_cap(t)     = φ_max · H(x(t)),     H(x) = 1/(1 + e^{−(x − x50)/s_diff})
```

Task difficulty is generated as d_i = x50 + s_diff·logit(u_i), so s_diff is in decades per logit unit. The current code marks the hardest-ranked tasks as non-automatable, which conflates compute difficulty with non-automatability (physical presence, liability, preference for humans); this is replaced. As a consequence, 1 − φ_max is a hard floor on the labor requirement; ℓ ≤ 0.02 requires φ_max ≥ 0.98.

**Proposition 2 (capability ceiling).** Covering a share q of automatable tasks requires x = x50 + s_diff·logit(q): 30.6 for q = 0.75, 32.3 for q = 0.90, 34.8 for q = 0.98 (S0). Under a binding power constraint,

```
x(t) ≤ log10( ς · util · s_T · (P̄(t)/PUE) · ε_max · Θ · Y_s · Ω_max )
```

The gap in decades between the required and the attainable x is the H2 test statistic.

### 4.4 Inference cost

```
e(i,t)    = e0 / Ω_inf(t) · 10^{θ·(d_i − x50)}               FLOP per worker-year of task output
c_AI(i,t) = r(t)/(util·Y_s) · e(i,t)                         $ per unit, at compute cost
```

The provider price p_AI,i (with markup μ) is set in §6.1. **Unit anchor:** e0 is re-anchored from ≈10⁸ tokens per worker-year × ≈10¹² FLOP/token plus agentic overhead, giving e0 ≈ 1e20 (range 1e19–1e21). The current e0 = 1e15 is about one H100-second and is 4–6 orders of magnitude too small.

**Proposition 2b (tail dominance).** With logistic difficulty and exponential inference cost, aggregate inference compute at full coverage is finite only if

```
k = θ · s_diff · ln 10 < 1
```

At S0, k = 3.45. At fixed Ω_inf, raising coverage of automatable tasks from 0.75 to 0.90 multiplies required inference compute by ≈24×; from 0.75 to 0.98 by ≈2000×. The last few percent of human work are disproportionately compute-expensive to replace. The result is sensitive to θ, which gets a dedicated sweep.

---

## §5 Production and labor (5.1–5.3 Recovered from `economy.py`; 5.4–5.6 New)

**Inputs:** c_AI, capable_i (§4); markups (§6); effective labor supply (§7). **Outputs:** w, Y, s_L, φ_eff, ℓ, E, ω_i*, K_d, flip set.

### 5.1 Task economy with conventional capital

```
Y   = A_Y(t) · K_o(t)^{α_K} · T^{1−α_K}
T   = [ Σ_i w_i · y_i^{(σ−1)/σ} ]^{σ/(σ−1)}      over task measure N(t)
```

Task price index P_T = [Σ_i w_i p_i^{1−σ}]^{1/(1−σ)}; numeraire P_Y = 1. K_o and A_Y grow at the no-AI rate g0. A_Y is calibrated so that Y(t0) = $105T and s_L(t0) = 0.55, implying a global average wage of ≈$16k per worker-year. The same model with φ ≡ 0 and I ≡ 0 is the **no-AI counterfactual** used in §8.

### 5.2 Automation rule

```
auto_i ⇔ capable_i ∧ min_n C_i(n) < w/γ_i
p_i    = C_i(n*_i)   if auto_i
       = w/γ_i       otherwise
```

C_i(n) is the oversight-inclusive AI cost from §5.4.

### 5.3 Proposition 3 (labor share)

Symmetric case (γ = 1, uniform AI price p_A):

```
s_L = (1 − α_K) · (N − φ)·w^{1−σ} / (φ·p_A^{1−σ} + (N − φ)·w^{1−σ})
```

σ < 1: s_L rises as p_A falls (Baumol). σ > 1: s_L falls toward zero. The current `Infeasible` exception for σ > 1 cannot occur once r clears the compute market (finite compute stock); a full run must never raise it.

### 5.4 Oversight and chunking

A task has J steps, executed by AI in n chunks with human verification at each chunk boundary; m = J/n steps per chunk.

- Per-step error: ε_i(t) = ε_H·e^{−(x(t) − d_i)/s_diff}. At parity (x = d_i) the AI error rate equals the human rate ε_H.
- Chunk error with Clayton dependence ψ across steps:
  ```
  e_m = 1 − (m·(1 − ε_i)^{−ψ} − m + 1)^{−1/ψ}      (ψ → 0: independent; ψ → ∞: comonotone)
  ```
- Detection probability D = q̄·(1 − ρ_conceal). Detected errors are redone; undetected errors cause damage κ_E,i × task value, compounded by (1 + λ) per downstream chunk.

```
R(n) = 1/(1 − D·e_m)                                                   rework multiplier
U(n) = Σ_{k=1}^{n} e_m·(1 − D)·(1 + λ)^{n−k}                           undetected damage index
C_i(n) = p_AI,i·R(n)  +  w·(ω_v + Γ_i·n/J)  +  (w/γ_i)·κ_E,i·U(n)
n*_i  = argmin_{n ∈ {1..J}} C_i(n),       ω_i* = ω_v + Γ_i·n*_i/J
```

Concealed errors (ρ_conceal) are never caught regardless of checking, so high-κ_E tasks remain human until AI reliability is high: a stakes-based floor on human work. Under extreme λ the optimum jumps to a corner (check every step, or do not automate): the step-6 test. `omega0 = 0.30` becomes a calibration target (observed oversight share at t0).

### 5.5 New tasks

dN/dt = ν·N. Each year's cohort of new tasks reuses the Sobol points with difficulty shifted so that d_new − x(t) has the same distribution as d − x(t0) had for original tasks. New tasks are therefore born human. ν is exogenous; endogenous task creation is deferred.

### 5.6 Labor supply and the four labor measures

```
L_s(w)  = L̄ · (w/w₀)^{ε_L}
L_eff   = L_s(w) − S(t) − Exits_cum(t)            (S, exits from §7)
w       ≥ w_floor = w_res_ratio · w₀              (if binding: employment rationed; excess → §7 inflow)
```

| Measure | Definition |
|---|---|
| Labor requirement ℓ(t) | (1 − φ_eff) + Σ_auto w_i·ω_i* — share of task-work still needing humans. **"1/50" ⇔ ℓ ≤ 0.02** |
| Labor share s_L | w·L_d / Y |
| Wage w | market-clearing (or floor) wage |
| Employment rate E | L_d / L̄ |

Labor market clearing: L_d = Σ_human w_i y_i/γ_i + Σ_auto w_i y_i ω_i* = L_eff.

---

## §6 AI industry (New)

**Outputs:** p_AI,i, Π_M, NPV_lab, F(t), I(t), the binding-constraint label, the rent ledger.

### 6.1 Pricing and the frontier sliver

Capabilities diffuse to competitors and open models after a lag τ_D. Tasks capable at x(t − τ_D) are **commoditized**: price = compute cost, zero markup. Tasks that became capable within the last τ_D years form the **frontier sliver**, where N_F providers compete in Cournot fashion subject to a limit price at the human cost:

```
p_AI,i = c_AI,i                                    if d_i ≤ x(t − τ_D)
p_AI,i = min((1 + μ_C)·c_AI,i, w/γ_i)              otherwise (frontier sliver)
μ_C    = 1/(N_F·σ − 1)                             requires N_F·σ > 1; else limit price binds
```

The task-level demand elasticity is σ, so no new parameter is needed; S0 gives μ_C = 0.5. The limit price means providers can never capture more than the wage of the human they replace.

### 6.2 Provider profit and Proposition 4

```
Π_M(t)  = Σ_{sliver} (p_AI,i − c_AI,i)·y_i  −  r·K_train  −  F_RD
NPV_lab = Σ_t (1 + ρ)^{−(t−t0)} · Π_M(t) / N_F
```

**Proposition 4 (rent dissipation).** Π_M < 0 whenever r·K_train + F_RD exceeds sliver rents, where the sliver measure is φ_max·[H(x_t) − H(x_{t−τ_D})]. **Limits (tests, step 8):** τ_D → 0 ⇒ Π_M = −(r·K_train + F_RD); N_F = 1 and τ_D → ∞ ⇒ monopoly pricing bounded by the limit price. Past the steep part of the logistic, the sliver thins and rents fall.

### 6.3 Investment and financing

```
I(t)    = min(I_demand, F(t), I_power)
F(t)    = F0·e^{g_F·τ}·e^{−ζ·L_cum(t)/F0} + X_H(t) + G(t) + max(0, Π_sector(t))
```

- I_demand: capacity to meet next period's expected compute demand at r_cost (adaptive expectations, §9.4) plus replacement of scrapped vintages; s_T fixed so training follows.
- I_power: investment that fits within power headroom P̄ − P_draw, including headroom freed by scrapping.
- L_cum: cumulative sector losses; ζ: investor patience; X_H: hyperscaler cross-subsidy (positive in S0); G: state support (0 in S0).

The model reports each year which constraint binds: **capability, power, or finance**. This label is the primary H2/H5 output.

**H5 loop:** losses → F ↓ → I ↓ → K ↓ → r ↑ → c_AI ↑ → φ_eff ↓ → displacement slows.

### 6.4 Rent ledger (to §8)

| Recipient | Flow |
|---|---|
| Chip vendors | R_H = (1 − 1/m_H)·I |
| Power/site owners | R_P = λ_P·P̄ |
| Model providers | Π_M |
| Compute owners | realized quasi-rents minus depreciation (includes unanticipated obsolescence losses) |
| Workers | wage bill minus no-AI wage bill |
| Consumers | residual, from the price index |

**Simplification:** s_T fixed; a race-driven endogenous training share is deferred.

---

## §7 Displaced-worker queue (Recovered from `cohorts.py`, with fixes; extensions New)

**Inputs:** labor released by flipping tasks and wage-floor rationing (§5). **Outputs:** S(t), exits, W, u, b(t), reemployment productivity, effective labor to §5.6.

The pool is a cohort array S[a] indexed by queue age a (quarters internally, reported in years), a ≤ A_max.

### 7.1 Inflow

```
λ(t) = max(0, labor released by flipped tasks − τ_turn·L̄) + wage-floor rationing
```

τ_turn = `turnover_ceiling` ≈ 0.025: retirements and redirected entrants absorb that much displacement without anyone entering the queue.

### 7.2 Capacity

```
μ(t) = μ(t−1) + (1/χ)·(λ(t − τ_c) − μ(t−1)),       μ(t0) = μ0
```

Capacity cost c_μ per slot-year is fiscal. Outflow is also bounded by openings in human tasks M(t) (new human-task jobs plus turnover openings): effective capacity μ_eff = min(μ, M).

### 7.3 Congestion: clearing function (replaces the current inflow-ratio factor)

```
X = μ_eff · S_eff / (S_eff + c2·μ_eff/h0),     S_eff = Σ_a S_a·e^{−ξ·a}
```

Outflow X is allocated across ages in proportion to S_a·e^{−ξ·a} (negative duration dependence). Because X depends on the employable stock S_eff rather than the raw stock, an aging pool clears more slowly; with ξ = 0, S_eff = S and the steady-state formula below holds exactly. Steady state with u = λ/μ_eff < 1 gives W = c2/(h0·(1 − u)); for c2 = 1 this equals Kingman's 1 + u/(1 − u) exactly, and for any c2 both share the heavy-traffic limit c2/(1 − u). For u ≥ 1 no steady state exists and the stock grows at rate λ − μ_eff. W is continuous and monotone in u. This fixes the defect in Appendix B (duration 2000 yr at u = 0.9995 but 1 yr at u = 1).

### 7.4 Scarring

- Permanent exit hazard π(a) = π_scar0 + π_scar_slope·a, applied as 1 − e^{−π·Δt}. Exits never return (hysteresis).
- Re-employment productivity loss: γ_reemp(a) = 1 − s_w0 − s_w1·a (Jacobson–LaLonde–Sullivan 1993; Davis–von Wachter 2011).

### 7.5 Fiscal flows

```
b(t) = b_rep · w_floor · S(t) + c_μ · μ(t)
```

Benefits are transfers in §8; lost output of the displaced is the real cost.

### 7.6 Proposition 5 (critical automation speed)

The queue is stable iff the displaced-labor flow stays below τ_turn + μ_eff: **≈4.5% of the labor force per year at S0.** Below it, W ∝ 1/(1 − u); above it, the stock grows and exits compound. The τ_c lag and flip bursts (§5.4 corner jumps; c2) create transient saturation even when the average flow is below threshold. The H4 test compares the simulated rate of labor-weighted task flips with this threshold.

### 7.7 Time resolution

Four quarterly sub-steps per annual period. `c2_burst` remains exogenous and swept; endogenizing it from flip timing is deferred.

---

## §8 Welfare ledger and verdicts (New)

### 8.1 Net social value

```
C(t) = Y − I − p_E·E − F_RD − c_μ·μ
NSV  = Σ_t β^{t−t0} · [C(t) − C_noAI(t) − SCC·Em(t)]  +  TV
TV   = [C(T) − C_noAI(T)] · β / (1 − β·(1 + g_T))       requires β(1 + g_T) < 1
```

Conventional investment in K_o is identical across worlds and cancels. NSV is reported with and without TV.

### 8.2 Discount-rate decomposition

NSV uses β = 0.97; NPV_lab uses ρ = 0.10. The ledger reports how much of any "socially good, privately bad" result comes from competitive dissipation (§6) versus the discount-rate gap, by recomputing NPV_lab at the social rate.

### 8.3 Distribution and consumption-equivalent welfare

Four groups: employed workers (net wages), displaced (benefits b), exited (minimum transfer), owners (population share n_K; all capital and rent income from §6.4). A proportional income tax balances the budget each period.

```
W          = Σ_t β^{t−t0} Σ_g n_g · c_g^{1−η}/(1 − η),     η = eta_ia = 1.5
1 + λ_CE   = (W_AI / W_noAI)^{1/(1−η)}
```

λ_CE is decomposed by Shapley value into four channels: **efficiency** (aggregate C), **inequality** (distribution across groups), **transition** (queue, exits, scarring), **environment** (carbon).

### 8.4 Hypothesis metrics

| | Deciding output |
|---|---|
| H1 | NPV_lab < 0 ∧ NSV > 0; PV rent shares by recipient; discount-rate contribution (8.2) |
| H2 | Year-by-year binding-constraint label; Proposition 2 gap in decades; energy per displaced worker-equivalent |
| H3 | Correlation and sign of dℓ vs ds_L, dw, dE across the σ sweep |
| H4 | Transition share of |λ_CE|; peak u(t), peak S(t), cumulative exits vs Proposition 5 |
| H5 | P(NPV_lab < 0 ∧ ℓ(2060) ≤ 0.02), with and without X_H and G |

### 8.5 Pre-registered verdict rules

Over the NROY draws of §10.4:

| Share of draws where the hypothesis holds | Verdict |
|---|---|
| ≥ 0.80 | Supported |
| ≤ 0.20 | Rejected |
| otherwise | Contested; report the 2–3 parameters with the largest total Sobol indices |

These thresholds are fixed before any simulation is run.

---

## §9 Numerical methods (9.1 Recovered; remainder New)

### 9.1 Task grid

Scrambled Sobol in 4 dimensions (d, κ_E, Γ, a), M = 4096 (power of two), equal weights N0/M, fixed ordering (never re-sorted; flip diagnostics track tasks by index). New-task cohorts reuse the points with shifted d and their own weight. **Convergence test:** key outputs at M = 4096 vs 16384 differ by < 0.5%; after it passes, sweeps may use M = 1024.

### 9.2 Period equilibrium

Within a period, compute supply is the **vintage merit order**: vintages sorted by operating (energy) cost. If demand is below capacity, r equals the operating cost of the marginal vintage; if demand exceeds capacity, r rises above the highest operating cost and the premium is a scarcity rent (to λ_P if power binds, otherwise to compute owners).

Nested root-finding with Brent's method (`scipy.optimize.brentq`):
- outer in w: P_Y(w) = 1 (monotone increasing in w);
- inner in r: K_d(w, r) = C_inf (monotone decreasing in r).

The chunking optimum is computed by vectorized enumeration over n ∈ {1..J} (an M×J array).

### 9.3 Per-period invariants (assertions)

- Σ w_i p_i y_i = P_Y·Y (relative error < 1e-8)
- L_d + S + Exits_cum = L_s (labor accounting)
- P_draw ≤ P̄ (1 + 1e-9)
- Σ income = Y (ledger closes)
- `Infeasible` never raised

On failure the run stops, reporting the parameter hash and period.

### 9.4 Expectations

I_demand extrapolates the trend of K_d over the previous 2 years (adaptive). A perfect-foresight variant is deferred (requires a path fixed point) and listed as a limitation.

### 9.5 Reproducibility and performance

Fixed seeds; every output carries a hash of its `Params`. Target < 2 s per run at M = 1024, enabling ≈10⁴ runs for global sensitivity with multiprocessing.

---

## §10 Calibration and validation (New)

### 10.1 Parameter classes

| Class | Parameters | Treatment |
|---|---|---|
| A. Measured | ε₀, κ₀, m_H, p_E, ι₀, SCC, Y₀, L̄, s_L0, P̄₀, PUE | Point value + CI from official or peer-reviewed source |
| B. Literature | σ, g_Ω0, g_eps, g_fab, ε_L, ξ_reemp, π_scar, s_w, μ0, δ_phys, ι_emb | Prior range from published estimates |
| C. Structural unknowns | φ_max, x50, s_diff, θ, ν, H, τ_Ω, g_Ωinf, τ_D, ζ, X_H | Wide priors; verdicts are decided here |
| D. Identities | A_Y, ς, `omega0` target, `g_kappa0` check | Solved, not chosen |
| N. Numerical settings | t0_year, T, M_tasks, sobol_seed, A_max, N0 | Fixed design choices, not calibrated |

### 10.2 Known discrepancies to resolve first

- g_Ω0: Ho et al. (2024) estimate effective-compute doubling ≈8 months (g ≈ 1.0/yr); S0 0.6 is conservative; prior [0.4, 1.2].
- g_eps: Epoch hardware price-performance doubling ≈2.5 yr (g ≈ 0.28) — consistent.
- g_Ωinf: Epoch reports 9–900×/yr price declines at fixed capability (combines Ω_inf, hardware, markup); S0 0.8 is at the low end; wide prior.
- P̄₀ = 15 GW: check against IEA *Energy and AI* (2025) for AI-specific servers; jointly determines ς (§4.2).
- ε₀: all-in server power, not chip TDP.
- e0: tokens-per-worker-year anchor (§4.4).
- x50, s_diff: calibrate H(x) against METR task-time-horizon series.

### 10.3 Backcast targets (2023–26)

| Target | Source | Grade |
|---|---|---|
| AI capex | Hyperscaler 10-K filings | A |
| Nvidia data-center revenue (↔ I, m_H) | 10-K filings | A |
| AI electricity use | IEA 2025; LBNL 2024 | A/B |
| Frontier training compute | Epoch database | B |
| GPU rental price (↔ r) | Market rental indices | B |
| API price at fixed capability | Epoch; Artificial Analysis | B |
| Lab revenue and losses | Disclosures and press | C |
| Early labor effects in AI-exposed occupations | Brynjolfsson–Chandar–Chen (2025) | B, low weight |

Grade C data enters with inflated observation variance: it constrains weakly and never pins a parameter alone.

### 10.4 Method: Bayesian history matching

Following Craig et al. (1997) and Vernon et al. (2010):

```
I(θ) = |z − f(θ)| / sqrt(Var_obs + Var_discrepancy)
```

Parameter vectors with I > 3 on any target are ruled out. The remainder is the NROY set. Draw ≈10⁴ vectors from NROY; compute Sobol/Saltelli indices; apply §8.5.

### 10.5 Out-of-sample check

Calibrate on 2023–25; predict 2026 targets; report errors before any verdict. A failed out-of-sample check must be stated ahead of results.

### 10.6 Provenance

`calibration/parameters.csv` with columns `name, value, lower, upper, prior, class, source, grade, notes`. `params.py` reads S0 values from it.

---

## §11 Update order, scenarios, build sequence

### 11.1 Per-period order

1. Technology: ε, κ, Ω, Ω_inf, P̄, ι, p_E (uses t−1 binding flag)
2. Capital: install I(t−1) vintage, decay, scrap; compute K, C_inf, C_train, P_draw
3. Capability: x, φ_cap, capable_i, e(i); frontier sliver uses x(t − τ_D)
4. Effective labor: L_eff (elastic part solved within step 5)
5. Equilibrium (w, r) with chunking → auto set, Y, K_d, ℓ, s_L, markups
6. Flips → released labor → §7 inflow; wage-floor rationing
7. Queue: 4 quarterly sub-steps; capacity update
8. Industry: Π_M, L_cum, F; I = min(I_demand, F, I_power)
9. Power binding flag, λ_P, E, Em
10. Ledger: C, rents, group consumption, tax rate; accumulate NSV, NPV_lab, W
11. New tasks: N(t+1)

No within-period feedback to an earlier step, so no within-year fixed point beyond §9.2.

### 11.2 Scenarios (diffs against S0)

| Name | Diff | Purpose |
|---|---|---|
| S0_baseline | — | Baseline |
| S1_motivating | φ_max = 0.98, ν = 0, σ = 1.2, eps_headroom = 10, τ_Ω = 4 | Strong-form thesis, tested not assumed |
| S2_scaling | eps_headroom = 300, τ_Ω = 20, g_P∞ = 0.08 | Scaling continues |
| S3_baumol | σ = 0.4, ν = 0.01 | Baumol dominates |
| S4_powerbound | g_P∞ = 0.02, g_P0 = 0.25 | Power binds |
| S5_finance_crunch | ζ = 2.0, X_H = 0 | H5: financing tightens |
| S6_supported | G = 0.3, p_E = 0.04, g_P∞ = 0.08 | H5: supported buildout |

### 11.3 Build sequence (each step gated by limiting-case tests)

| Step | Work | Gate |
|---|---|---|
| 0 | Fixes: units, e0, ε₀/κ₀/m_H, clearing-function queue, 4th Sobol dimension, Ω_inf decay, `parameters.csv` | Existing tests updated; queue continuity at u ∈ {0.999, 1.0, 1.001}; Kingman values (1.33, 2, 4, 20) at c2 = 1 |
| 1–2 | Outer Cobb–Douglas nest | No-AI s_L = 1 − α_K; Proposition 3 × (1 − α_K) |
| 3 | Equilibrium with merit order | Regime switch at K_d = C_inf; `Infeasible` never raised |
| 4 | Vintages, scrapping | Proposition 1; Hall limit at p_E = 0 |
| 5 | Capability and power | Proposition 2; Proposition 2b ratios ≈24× and ≈2000× |
| 6 | Chunking | Corner solutions at extreme λ; ψ → 0 gives independent product |
| 7 | Queue, scarring, fiscal | Proposition 5 threshold; W continuous at u = 1 |
| 8 | Industry | Proposition 4 limits (τ_D → 0; monopoly) |
| 9 | Ledger and no-AI counterfactual | φ ≡ 0 ⇒ NSV = 0, λ_CE = 0 |
| 10 | Calibration, backcast, global sensitivity | §10.4 NROY non-empty; §10.5 out-of-sample report |

Tests follow the existing style: pytest, closed-form limiting cases, `test_<step>_<case>` naming.

---

## Appendix A — Parameters

### A.1 Existing parameters: changes

| Parameter | Current | Change |
|---|---|---|
| m_H | 3.0 | → 4.0 (75% gross margin; comment was inconsistent) |
| eps0 | 1.4e15 | → 0.75e15 provisional (all-in per kW); final from §10 |
| kappa0 | 3.0e-11 | → 4.0e-11 provisional ($30k per 0.75e15 FLOP/s unit); final from §10 |
| e0 | 1.0e15 | → 1.0e20 (worker-year anchor), range [1e19, 1e21] |
| g_kappa0 | 0.30 | free parameter → calibration check |
| x_t0 | 26.0 | free parameter → calibration target pinning ς |
| omega0 | 0.30 | hook → calibration target |
| g_Omega_inf | 0.80 | now decays with τ_Ω |

### A.2 New parameters (provisional S0 and prior range; final values from §10)

| Name | Meaning | S0 | Range | Class |
|---|---|---|---|---|
| alpha_K | Conventional capital share | 0.40 | [0.30, 0.45] | A |
| varsigma | Frontier-run share of training compute | derived | — | D |
| J_steps | Steps per task | 20 | [5, 50] | C |
| eps_H | Human per-step error rate | 0.01 | [0.002, 0.05] | B |
| omega_v | Verification labor per unit AI output | 0.05 | [0.01, 0.20] | C |
| tau_D | Capability diffusion lag (yr) | 1.0 | [0.25, 3.0] | C |
| g_F | Trend growth of external finance | 0.05 | [0.0, 0.15] | C |
| zeta | Investor loss sensitivity | 0.5 | [0.0, 2.0] | C |
| X_H0 | Hyperscaler cross-subsidy ($T/yr) | 0.30 | [0.0, 0.6] | C |
| G0 | State support ($T/yr) | 0.0 | [0.0, 0.5] | C |
| F_RD | Provider fixed cost ($T/yr) | 0.02 | [0.005, 0.05] | B |
| iota_emb | Embodied emissions (tCO₂ per $ hardware) | 1.0e-4 | [3e-5, 3e-4] | B |
| tau_C | Carbon price ($/tCO₂) | 0.0 | [0, 190] | scenario |
| c_mu | Retraining cost per slot-year (× w₀) | 0.5 | [0.2, 1.5] | B |
| s_w0 | Re-employment productivity loss, base | 0.10 | [0.05, 0.20] | B |
| s_w1 | Additional loss per year queued | 0.02 | [0.0, 0.05] | B |
| b_rep | Benefit replacement ratio | 0.40 | [0.2, 0.7] | A |
| n_K | Owner population share | 0.10 | [0.05, 0.20] | A |
| g_T | Terminal growth for TV | = g0 | [0.0, 0.03] | C |
| h0_reemp | Base re-employment hazard (/yr); W = c2/(h0(1 − u)) | 1.0 | [0.5, 3.0] | C |

---

## Appendix B — Code defects found in review (fixed in build step 0)

1. **Queue discontinuity at saturation** (`cel/cohorts.py:61`). The congestion factor is a function of inflow ratio u and is set to 1 for u ≥ 1. Measured steady-state durations: 2000 yr at u = 0.9995, 1.0 yr at u = 1.0, 11 yr at u = 1.005. Fixed by §7.3.
2. **Markup comment** (`cel/params.py:29`). m_H = 3 implies 67% gross margin, not 75%.
3. **Unit inconsistency** (`cel/params.py:24,27`). ε₀·unit_kW = 1.4e15 FLOP/s per unit at κ₀ gives $42k per unit; the comment assumes $30k per 1e15.
4. **Money-unit mismatch** (`cel/economy.py`). c_AI is in $ while w and Y are normalized; `Y0_trillion` is never used. Fixed by the worker-year unit (§0) and A_Y calibration (§5.1).
5. **Unbounded inference efficiency** (`cel/grid.py:62`). Ω_inf grows without bound; fixed by §4.1.
6. **Non-automatability conflated with difficulty** (`cel/economy.py:62`). `rank < phi_cap` makes the hardest tasks the non-automatable ones; fixed by §4.3.

---

## Appendix C — References

- Acemoglu, D. & Restrepo, P. (2018). The Race between Man and Machine. *AER*.
- Acemoglu, D. (2024). The Simple Macroeconomics of AI. *NBER WP*.
- Autor, D., Chin, C., Salomons, A., Seegmiller, B. (2024). New Frontiers: The Origins and Content of New Work. *QJE*.
- Brynjolfsson, E., Chandar, B., Chen, R. (2025). Canaries in the Coal Mine? Stanford Digital Economy Lab WP.
- Craig, P., Goldstein, M., Seheult, A., Smith, J. (1997). Pressure matching for hydrocarbon reservoirs. *Case Studies in Bayesian Statistics*.
- Davis, S. & von Wachter, T. (2011). Recessions and the Costs of Job Loss. *BPEA*.
- Hall, R. (1968). Technical Change and Capital from the Point of View of the Dual. *REStud*.
- Ho, A. et al. (2024). Algorithmic Progress in Language Models. Epoch AI.
- IEA (2025). *Energy and AI*.
- Jacobson, L., LaLonde, R., Sullivan, D. (1993). Earnings Losses of Displaced Workers. *AER*.
- Karmarkar, U. (1989). Capacity Loading and Release Planning with WIP and Leadtimes. *J. Manufacturing and Operations Management*.
- Kingman, J. (1961). The single server queue in heavy traffic. *Proc. Cambridge Phil. Soc.*
- LBNL (2024). *United States Data Center Energy Usage Report*.
- Lucas, R. (1987). *Models of Business Cycles* (consumption-equivalent welfare).
- METR (2025). Measuring AI Ability to Complete Long Tasks.
- Saltelli, A. et al. (2010). Variance based sensitivity analysis of model output. *Computer Physics Communications*.
- Vernon, I., Goldstein, M., Bower, R. (2010). Galaxy formation: a Bayesian uncertainty analysis. *Bayesian Analysis*.
