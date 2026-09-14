# Opacity mode (c-k vs lbl) for low-resolution instruments

*Chantier A Phase 4 — see `Notes/plan_revision_starships.md` for the full implementation
history/design discussion. This doc explains the feature itself: what it does, and what
each combination of options actually gets you.*

## The problem this solves

`petitRADTRANS` can generate a model spectrum two ways:

- **`c-k`** (correlated-k): coarse, pre-binned opacities. Fast, low memory. Used by default
  for every low-resolution instrument (`spectrophotometric_data`, `photometric_data`).
- **`lbl`** (line-by-line): full-resolution opacities. Much slower and more memory-hungry,
  but physically more accurate — this is what every high-resolution instrument (SPIRou,
  NIRPS, ...) already uses, since c-k isn't fine enough to resolve individual lines at
  R ~ 60,000+.

For a low-resolution instrument with fine enough native resolution or wide enough
wavelength coverage (e.g. JWST/NIRSpec G395H, R ~ 700 but spanning a wide swath with real
line structure), c-k opacities can be too coarse to model the data well. `opacity_mode`
lets you ask for `lbl` opacities on a **per-instrument** basis, so you only pay the extra
cost where it actually matters.

## The YAML key

```yaml
spectrophotometric_data:
  g395H_1:
    file_path: '...'
    file_name: '...'
    opacity_mode: 'lbl'   # or 'c-k' (default) -- optional, per instrument
```

- Only recognized under `spectrophotometric_data`. There is **no** `opacity_mode` key
  for `photometric_data` — see [Coverage decides everything](#coverage-decides-everything)
  below for why that's not a limitation in practice.
- Default is `'c-k'` for every instrument that doesn't set it — existing configs are
  completely unaffected by this feature.

## Coverage decides everything

Under the hood, `opacity_mode: 'lbl'` doesn't create a separate "low-res lbl" model. It
simply folds that instrument's wavelength range into the **high-resolution** wavelength
range (`wv_range_high`) — exactly as if it were another real high-resolution instrument.
From that point on, the existing high/low dispatch (`assign_model_type`, already in place
before this feature existed, for e.g. WFC3 data that happens to fall inside a SPIRou
window) takes over automatically:

- An instrument is modelled with the (lbl) high-res model if its **entire** wavelength
  range is covered by `wv_range_high` — whether that coverage comes from a real
  high-resolution instrument's range, another instrument's `opacity_mode: 'lbl'` range, or
  both merged together.
- Otherwise, it falls back to the shared c-k low-res model (`wv_range_low`).

This is why `photometric_data` doesn't need its own `opacity_mode` key: a photometric
band that happens to land entirely inside an lbl-extended `wv_range_high` window
automatically gets lbl-quality treatment for free, with zero extra configuration.
Similarly, a `spectrophotometric_data` instrument left at the default `'c-k'` can *still*
end up modelled with lbl opacities, if another instrument's `opacity_mode: 'lbl'` range
happens to cover it — again, nothing to configure, it just falls out of the coverage
check.

**Partial coverage never mixes modes within one instrument.** If an instrument's range
only partially overlaps `wv_range_high` (even by a little), the whole instrument falls
back to c-k — never half-lbl, half-c-k for the same dataset.

## Worked scenarios

### 1. JR, real high-res instrument, all spectrophotometric data at c-k (baseline)

```yaml
retrieval_type: 'JR'
instrum: [spirou]
spectrophotometric_data:
  wfc3: {...}   # opacity_mode not set -> 'c-k'
```

Unchanged behaviour from before this feature existed. `wfc3` gets the dedicated c-k
low-res model (or, if its range happens to already be covered by SPIRou's window, the
reused high-res model — that reuse mechanism predates `opacity_mode` entirely).

### 2. JR, real high-res instrument, one spectrophotometric instrument in lbl

```yaml
retrieval_type: 'JR'
instrum: [spirou]
spectrophotometric_data:
  g395H_1: {opacity_mode: 'lbl'}
  g395H_2: {opacity_mode: 'lbl'}
```

The KELT-20b test case this feature was built and validated against (SPIRou + JWST/NIRSpec
G395H NRS1/NRS2). `g395H_1`/`g395H_2`'s ranges get folded into `wv_range_high` alongside
SPIRou's own range — 3 disjoint windows in this real case (G395H is far outside SPIRou's
near-IR coverage), so 3 separate `petitRADTRANS` objects get initialized. Real HIGH RES
likelihood (SPIRou's actual exposures) is untouched — this only affects how the
spectrophotometric data's synthetic model is generated.

### 3. Pure LRR, c-k only, no real high-res instrument

```yaml
retrieval_type: 'LRR'
instrum: []
spectrophotometric_data:
  wfc3: {...}   # opacity_mode not set -> 'c-k'
```

The original, long-standing use case — a retrieval on low-resolution data alone. Nothing
new here.

### 4. Pure LRR, lbl only, no real high-res instrument

```yaml
retrieval_type: 'LRR'
instrum: []
spectrophotometric_data:
  g395H_1: {opacity_mode: 'lbl'}
  g395H_2: {opacity_mode: 'lbl'}
```

A retrieval using **only** lbl-flagged low-resolution data, with no real high-resolution
spectrograph anywhere in the run. Validated on the same KELT-20b G395H data as scenario 2,
`instrum: []` instead of `[spirou]`. A few things fall back to sensible defaults in this
case, all covered by unit tests and validated with real `petitRADTRANS` on Narval:

- The reference resolution used to generate the lbl model (normally the finest real
  high-res instrument's resolving power) falls back to the model's own native lbl
  sampling resolution (`get_res_instru`).
- The representative orbital phase used for any region-combination kernel (see below)
  falls back to the ephemeris (mid-eclipse/mid-transit), the same convention already used
  for the c-k low-res path — there's no real per-exposure high-res visit to pull a phase
  from.

### 5. Mixed lbl + c-k within the same run

Nothing stops an instrument list from mixing `opacity_mode: 'lbl'` and `'c-k'` (or
unset) freely — each instrument's own coverage against `wv_range_high` decides its fate
independently. There's no requirement that all low-res instruments share the same mode.

## Multi-region vs single-region

Low-resolution data (spectrophotometric or photometric) has no real per-exposure time
series of its own — it's integrated over a whole visit. This matters when `region_id`
has more than one entry (e.g. a citrus/longitude-split model): instead of combining
regions once per exposure (the way real high-resolution data does), the model is
evaluated at a handful of **representative phases** (`representative_phases_low` — 4
phases spread across transit, or 2 just before/after eclipse, computed once from the
ephemeris in `setup_retrieval`; override with `representative_phases_low` or
`n_phases_low` in the YAML) and the resulting spectra are averaged.

This applies identically whether the underlying model comes from the dedicated c-k
low-res atmo objects or the (possibly lbl) reused high-res ones — same averaging
mechanism, same representative phases, regardless of `opacity_mode`. For a single region
(no `get_ker`/region combination needed at all), this whole averaging step is skipped and
the model is generated directly — cheaper, and exactly equivalent since there's nothing
to combine.

The same single entry point (`retrieval.prepare_static_model`) is used both inside
`lnprob` (to fit) and by `retrieval_utils.get_contribution` (to build post-retrieval
species-contribution plots) — so a contribution plot always reflects the exact model the
retrieval actually fit against, regardless of region count or opacity mode.

## Cost

`lbl` is expensive — that's the whole reason `opacity_mode` is opt-in per instrument
rather than a single global flag. Concretely, on the KELT-20b G395H test case (see
`Notes/plan_revision_starships.md` for the full numbers):

- The `petitRADTRANS` opacity tables are loaded once per run (not once per MCMC
  iteration) — a real high-res + 2 lbl bands setup took ~8 minutes the first time, then
  ~38 s per further `lnprob()` call.
- Adding lbl bands roughly doubled the per-iteration cost compared to the same run
  without them (real high-res data alone).
- Memory is **not** duplicated per MCMC worker: `retrieval.py::prepare_run`'s existing
  pre-run sanity check already calls `lnprob()` once in the parent process before the
  `multiprocessing.Pool` is created, so forked workers inherit the already-initialized
  `petitRADTRANS` objects via copy-on-write instead of re-loading them independently.

## Known limitations

- A custom, phase-dependent `get_ker` (rotation/citrus kernel) combined with a run that
  has **no real high-resolution visit at all** (scenario 4) only ever sees the
  ephemeris-averaged representative phases, never a genuine per-exposure phase — there is
  none to give it. This is the same limitation the c-k low-res path has always had; lbl
  doesn't make it any better or worse.
- Real `petitRADTRANS` validation of this feature so far is limited to the KELT-20b
  dayside dataset (SPIRou + JWST/NIRSpec G395H). The mechanics are unit-tested more
  broadly, but a new dataset/instrument combination is worth a real Narval sanity check
  before trusting it blindly.
