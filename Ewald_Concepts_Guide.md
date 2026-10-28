# Ewald Summation — A Concepts Guide

This is the companion to the ion stage of the simulation: the theory of the
`LennardJonesEwald` force kernel, written so you can implement it yourself.
No code — just the idea, the math, and the pitfalls. Everything is in the
reduced units the engine already uses (lengths in the reference σ, energies in
ε_ref, k_B = 1).

---

## 1. The problem: Coulomb doesn't like boxes

The engine is periodic: every particle interacts not only with the particles
in the box but with all their infinite periodic image copies. For the
Lennard-Jones potential this is a non-issue — the force dies like 1/r⁷, so
beyond a cutoff of a few σ nothing measurable is left, and the Verlet list
simply ignores the rest.

Coulomb is different. The electrostatic energy of the periodic system is

```
E = (k_e / 2) · Σᵢ Σⱼ Σₙ' qᵢ qⱼ / |rᵢⱼ + n·L|
```

where `n·L` runs over all image-cell offsets and the prime means "skip the
self term" (i = j, n = 0). This sum is **conditionally convergent**: the
terms fall off like 1/r, but the number of terms per spherical shell grows
like r², so each shell contributes ~1/r — the sum does not converge
absolutely, and *its value depends on the order in which you add the
shells*. There is no single "natural" answer. Consequences:

- **Cutting it off** (the SF-cutoff approach) works only when the system is
  neutral and charges are locally mixed, so distant shells mostly cancel.
- **If the total charge Q ≠ 0**, the sum outright diverges. A "true" answer
  for a net-charged periodic cell only exists after you declare what
  neutralizes it. The standard declaration — used by every quantum-chemistry
  and MD code — is a **uniform neutralizing background**: a smeared charge
  −Q filling the box. This is called the *tin-foil* convention.
- **Just summing N² minimum-image pairs** is not a third option — it
  silently drops all images, which for Coulomb are not negligible. It is
  neither the cutoff answer nor the periodic answer.

So "true no matter whether the system is neutral" means: the exact energy
and forces of the periodic problem under the standard background
convention. That is exactly what Ewald summation computes.

---

## 2. The trick: split by smoothness

Ewald's idea is to split each 1/r into a short-ranged piece and a smooth
piece, and to compute each piece in the domain where it is cheap:

```
1/r  =  erfc(αr)/r  +  erf(αr)/r
        ──────────     ─────────
        short-ranged   smooth, long-ranged
        → real space  → Fourier (reciprocal) space
```

The physical picture: dress every point charge qᵢ with a diffuse Gaussian
"screening cloud" of total charge −qᵢ centered on it.

```
cloud density:   ρᵢ(r) = −qᵢ · (α³/π^1.5) · exp(−α²|r − rᵢ|²)
```

The point charge and its own cloud together form a compact, **neutral**
object, and the interaction between two such objects is precisely the erfc
term: point charges screened by clouds interact through `erfc(αr)/r`, which
dies off extremely fast (faster than any power of r). That part is
short-ranged — perfect for the Verlet lists, exactly like LJ.

But adding the clouds changed the system. We must add back the interaction
of the clouds *among themselves*. The cloud-cloud interaction per pair is
the erf term: `erf(αr)/r` (this is a fact worth internalizing: the
electrostatic potential of a Gaussian charge distribution is erf(αr)/r).
That leftover is **smooth and gently varying** — and smooth periodic
functions have sparse Fourier series, because a Gaussian's Fourier
transform is itself a Gaussian. So the leftover is cheap to sum over all
images at once in reciprocal space.

The parameter **α** (dimensions 1/length) sets the cloud width: large α →
tight clouds → real part dies quickly (cheap real space) but the leftover is
sharper (more k-vectors needed); small α → the opposite. α does not change
the physics — it only moves work between the two sums. That property is the
basis of the strongest correctness test (Section 7).

Sanity limits: α → 0 makes erfc → 1 (everything real-space bare Coulomb,
reciprocal → 0); α → ∞ pushes everything into the reciprocal sum. Same
physics either way.

---

## 3. The four pieces

The total Coulomb energy is

```
E_coulomb = E_real + E_recip + E_self + E_background
```

`k_e` is the Coulomb constant in your reduced units (Section 5), V = L³ the
box volume. Below, `rᵢⱼ = |rᵢ − rⱼ|` with minimum-image wrapping, and
`r̂ᵢⱼ = (rᵢ − rⱼ)/rᵢⱼ`.

### 3.1 Real space (pairs — goes through the Verlet lists)

```
E_real = k_e · Σ_{i<j} qᵢ qⱼ · erfc(α rᵢⱼ) / rᵢⱼ        (pairs within r_cut)
```

The pair force on particle i, along the pair axis:

```
F_real(i ← j) = k_e qᵢ qⱼ · [ erfc(αr)/r²  +  (2α/√π) · exp(−α²r²)/r ] · r̂ᵢⱼ
```

**The bracket has two terms and both are mandatory.** The second term is
the force contribution of the screening cloud's own field (it is the
derivative of erf(αr)/r), and forgetting it is *the* classic Ewald bug: the
result still looks plausible, still conserves energy, and is quietly wrong
by ~10%. Both terms are positive-definite, so like charges repel in the
real part and opposites attract — same sign conventions as the LJ term, so
it drops straight into the pair kernel's "magnitude × axis" pattern.

Two practical notes:

- **Truncation is legitimate here.** erfc(αr) is genuinely dead at the
  cutoff: erfc(3.5) ≈ 7×10⁻⁷, so with α·r_cut ≈ 3.5 the truncated tail is
  below single-precision noise. No shifted-force trick is needed — the
  screening *is* the smooth cutoff function.
- **Minimum image suffices.** The first neglected image shell sits at
  distance ≈ L − r_cut, where erfc is astronomically small. So the real
  part uses the same minimum-image displacement the engine already
  computes for LJ.

### 3.2 Reciprocal space (global — once per step, no neighbor lists)

The allowed wavevectors are set by the box:

```
k⃗ = (2π/L) · (n₁, n₂, n₃)     with integer n's,  n ≠ 0
```

For each wavevector define the **structure factor** — the charge-weighted
Fourier amplitude of the configuration:

```
S(k⃗) = Σⱼ qⱼ · exp(i k⃗·rⱼ)     (equivalently: Σ qⱼ cos(k⃗·rⱼ) + i Σ qⱼ sin(k⃗·rⱼ))
```

Then

```
E_recip = (2π k_e / V) · Σ_{k⃗≠0}  |S(k⃗)|² · exp(−k²/4α²) / k²
```

and the force on particle i:

```
F_recip(i) = (4π k_e / V) · qᵢ · Σ_{k⃗≠0} (k⃗/k²) · exp(−k²/4α²) · Σⱼ qⱼ sin(k⃗·(rᵢ − rⱼ))
```

Reading the force formula: for every wavevector, add a vector pointing
along k⃗, weighted by how "in phase" particle i is with the charge
distribution, screened by the Gaussian factor exp(−k²/4α²). It is a global
quantity — every particle's force depends on every other particle's
position — which is why this piece cannot be a pair kernel and needs its own
once-per-step pass.

Essential details:

- **The k = 0 term is excluded, always.** k = 0 is the infinite uniform
  self-field of the system; dropping it is mathematically identical to
  adding the uniform neutralizing background. This is why the exact same
  formulas are valid for neutral and net-charged systems: the convention is
  built in. (For a neutral system S(0) = Σqⱼ = 0 anyway — the two
  viewpoints agree.)
- **exp(−k²/4α²)/k² depends only on α and the box** — precompute it once
  per k-vector at setup (along with the k⃗ list). Per step you only
  recompute S(k⃗) and the forces, cost ~ N × K (N particles, K
  wavevectors).
- **±k symmetry halves the work**: E is even in k⃗, F is odd. Summing over
  "half the sphere" and adding both signs is a free 2× saving.
- **No minimum-image wrapping needed here.** Shifting a position by a box
  length changes k⃗·rⱼ by 2π × integer, which leaves exp(i k⃗·rⱼ) and the
  sine unchanged. Raw positions go in.

### 3.3 Self term (energy only)

The reciprocal sum, written through |S|², accidentally includes the
interaction of each charge with *its own* cloud (the i = j diagonal). That
is not physical and must be subtracted:

```
E_self = −k_e · (α/√π) · Σᵢ qᵢ²
```

It is a constant (depends only on charges and α) — **it exerts no force**
and never affects the dynamics. It only matters when you evaluate energies:
the Madelung test and the finite-difference force test both need it.

### 3.4 Background term (energy only, net-charged systems)

If the total charge is Q ≠ 0, the tin-foil convention adds one more
constant: the interaction of the neutralizing background with the clouds,
plus the background with itself:

```
E_background = −k_e · π Q² / (2 α² V)
```

Again: **zero force** — the background is uniform, it pushes on nothing.
It exists so that energies of net-charged cells are the standard values.
If you ever need single-precision energies of *small* net-charged cells,
the Makov–Payne finite-size correction is the next refinement (also
energy-only); for this project it is optional.

---

## 4. Choosing α, r_cut, and k_max

The three knobs, with their error consequences (precise bounds: the
Kolafa–Perram formulas; these rules of thumb are enough in practice):

```
real-space truncation error   ~ exp(−(α · r_cut)²)
reciprocal truncation error   ~ exp(−k_max² / (4α²))
```

where k_max is the largest |k⃗| you include. The recipe:

1. Pick the real-space cutoff r_cut from the engine's constraints (cell
   list needs r_cut + skin ≤ L/3). Then choose α from your error target:
   α = 3.5/r_cut gives ~10⁻⁶ relative force error; α = 3/r_cut gives
   ~10⁻⁴, which is plenty for MD.
2. Match the reciprocal accuracy: k_max ≈ 7α (for the 10⁻⁶ level) or
   k_max ≈ 6α (for the 10⁻⁴ level). In integer counts per axis:
   n_max = k_max·L/(2π), rounded up.

Worked numbers for the demo box (L = 16, reference σ = 1):

| α·r_cut | r_cut | α | k_max | n_max | wavevectors (sphere) | ≈ rel. error |
|---|---|---|---|---|---|---|
| 3.0 | 4.5 | 0.67 | 4.0 | 10 | ~4×10³ (~2×10³ with ±k) | ~10⁻⁴ |
| 3.5 | 4.5 | 0.78 | 5.5 | 14 | ~1.1×10⁴ (~5×10³ with ±k) | ~5×10⁻⁶ |

At N ≈ 1000 particles, the looser row costs ~2×10⁶ trig evaluations per
step — visible on CPU (milliseconds), trivial on GPU. Start there, tighten
only if a test demands it. α shifts work between the two sums; the total
result must not depend on it (Section 7's first test).

---

## 5. Units in this codebase

Everything above is in the engine's reduced units, so the only new constant
is **k_e**. Fix it through the Bjerrum length — the separation at which the
Coulomb attraction between unit charges equals thermal energy:

```
λ_B = k_e q² / (k_B T)   →   k_e = λ_B · T*   (charges ±1, k_B = 1)
```

with T* the reduced temperature the engine already uses. λ_B ≈ 1–2 gives
lively ion dynamics (pairs form and break on screen); real molten salts are
far more strongly coupled and would look frozen. Then:

- α is in 1/[length] (i.e. 1/σ): ~0.7–0.8 for the numbers above,
- k⃗ is in 1/[length] via k = 2π·n/L,
- all four energy formulas carry the same k_e factor,
- energies come out in ε_ref — × ε_ref_kelvin converts to Kelvin, exactly
  like every other energy in the engine.

Net charge is now *legal by design* (the background term is the bookkeeping
for it), so the constructor's neutrality validation turns into at most an
informational note.

---

## 6. How it maps onto the engine

The kernel grows a second phase — this is the one architectural change:

```
pair phase (existing Verlet-list loop, per pair):
    LJ term (unchanged) + Ewald real-space term
    → same "scalar magnitude × pair axis" pattern, same lists

global phase (new, once per step, after the pair loop):
    reciprocal forces for every particle, added into the same
    force accumulator; then acceleration()/velocity as usual
```

- The LJ variant's global phase is simply empty — the interface is shared.
- The enum variant carries its settings: α, k_max, and the precomputed
  k⃗/exp(−k²/4α²) table.
- The thermostat, integrator, PBC, renderer, cell/Verlet lists: all
  untouched.
- Bookkeeping warning for energies: the Verlet lists store every pair
  *twice* (once in each partner's list). Forces are fine as-is, but any
  energy sum over list entries counts each pair twice — halve it.
- Precision: compute the S(k⃗) sums (Σ qⱼ cos, Σ qⱼ sin over all
  particles) in f64, cast forces back to f32 at the end. f32 accumulation
  over 1000 particles loses several digits and the reciprocal term will
  visibly jitter.

---

## 7. How to know it's right

Ordered from "structural" to "physical":

1. **α-invariance (the signature test).** Fix a configuration. Evaluate
   total energy/forces at α = 0.6, 0.8, 1.0, 1.2 — re-tuning r_cut/k_max
   per the error rules so each evaluation is converged. All must agree.
   Ewald's answer is *defined* by this independence; any error in any term
   almost always breaks it.
2. **The Madelung test.** Place ±q alternately on the simple-cubic lattice
   (checkerboard parity) — that is the CsCl-type ionic crystal, whose
   tabulated Madelung constant is α_M ≈ 1.76267. With lattice spacing s
   (and **an even number of sites per axis**, or the periodic boundary
   puts like charges next to each other), the exact Coulomb energy per ion
   pair is

   ```
   E_pair = −α_M · k_e q² / s
   ```

   LJ switched off. Plot your total E_coulomb against α: it should wobble
   at small α (truncation) and then plateau onto the exact value to your
   chosen accuracy. This single test validates real + reciprocal + self +
   background and every exponent.
3. **Finite differences vs analytic forces.** Numerically differentiate
   the *total* energy (all four pieces, including self — it's constant, so
   harmless) by displacing one particle; compare with the force routine.
   This is the test that catches the missing Gaussian term in the
   real-space force and any sign slip in the reciprocal gradient.
4. **Momentum.** Σ F must be 0 for *any* configuration, including
   net-charged ones. (Real part is equal-and-opposite by construction;
   reciprocal cancels by the antisymmetry of sin; self/background exert
   nothing.)
5. **Net-charge sanity.** A few extra like charges in an otherwise neutral
   fluid must not cluster onto each other — like-charge clustering is the
   classic cutoff failure that Ewald exists to fix.
6. **SF cross-check.** In the neutral, well-mixed regime the old
   SF-cutoff variant should agree qualitatively (same pair statistics,
   same structure). The residual difference *is* the cutoff bias you were
   paying — now you can measure it.

---

## 8. Pitfalls checklist

1. Real-space force missing its second term, (2α/√π)·exp(−α²r²)/r — the
   most common Ewald bug. (Test 3 catches it; test 2 does too.)
2. Forgetting E_self (or the background term) in energies — forces are
   unaffected, but Madelung and finite-difference tests will be off by a
   constant-looking-but-not-constant amount.
3. Including k = 0 — divergent. Its exclusion is not an approximation; it
   *is* the neutralizing-background convention.
4. Double-counted pair energies from the two-sided Verlet lists (halve
   them; forces are fine).
5. Wrong exponents: the screening factors are exp(−α²r²) in real space,
   exp(−k²/4α²) in reciprocal space. The α-swap test detects any mix-up.
6. Forgetting minimum image in the real part (needed) — or applying it in
   the reciprocal part (harmless but wasted work; raw positions are
   fine).
7. f32 accumulation of S(k⃗) — use f64 accumulators internally.
8. Truncating the real sum at r_cut while α is too small: verify
   α·r_cut ≳ 3 so the kernel is actually dead at the cutoff (unlike the
   SF kernel, there is no sign-flip hazard beyond it — the tail is simply
   dead).
9. Lattice tests with an odd number of sites per axis — the periodic
   continuation breaks the checkerboard and the Madelung reference is
   wrong, not your Ewald.
10. Units: k_e must be the same reduced-unit constant in all four
    formulas; α in 1/σ; k⃗ = 2π·n/L with L the *reduced* box length.

---

## 9. Where this goes next

- **GPU:** the real-space term joins the force shader's variant switch
  like any other kernel. WGSL has no erfc — use a documented rational
  approximation of erfc (the classic Abramowitz–Stegun formula 7.1.26 is
  ~10⁻⁷ accurate, below your truncation error, and is a handful of
  arithmetic ops). The reciprocal part becomes one small second compute
  pass (particles in parallel, k-loop inside). PME/FFT remains unnecessary
  until N grows by orders of magnitude.
- **Convention coherence:** the tin-foil background is the *same*
  convention every quantum-chemistry code uses for charged cells — so your
  future RHF reference data (and the NN trained on it) and this classical
  tier all agree on what a net-charged periodic cell means.
- The SF-cutoff variant stays alongside as the cheap cousin: GPU-trivial,
  and now a measurement stick for the cutoff bias.
