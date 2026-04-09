# 1. Introduction

- Broad objective: I want to study hadronic Physics at low energies, particularly pion-pion scattering
- Precise objective: reduce the truncation error in the partial wave expansion of T by calculating up to G-wave contributions + study the stability / convergence of T when changing the wave at which I truncate

- Problem: QCD is non-perturbative in this regime (talk about confinement)

- Solutions:
    1. LQCD: very resource-expensive
    2. ChiPT: you need to choose an order in perturbation theory and calibration is nontrivial
    3. Dispersive methods: you start with some basic axioms and then build from consistency --> model independent!

# 2. An intro to scattering

- S matrix in QM and QFT
- Relationship with invariant amplitude (remark this can't be obtained with perturbative QCD in this regime)
- Mandelstam variables and on-shell condition
- Link: we shall now see T obeys a so-called dispersion relation

# 3. An intro to dispersive methods

- First dispersion relations: Kramers-Krönig
- They first appeared in the 60s, before we had a theory for the strong interaction, and have recently gained attention again because one can use them to verify whether the chiral condensate is indeed the LO term in ChiPT (it is) and also useful as a bridge between theory and experiment
- Also interesting on their own: they allow us to test whether the first principles we believe to be true actually hold
- What is a dispersion relation?: introduce close relationship between causality and analyticity as in Causality and Dispersion Relations
- Titchmarsh theorem and Plemelj formulae
- Explain subtraction terms (Causality and DRs)
- Deriving dispersion relation for T (until 1.131 in thesis)
    - Imposing unitarity
    - Imposing Mandelstam hypothesis --> explain poles in real axis and branch cuts
    - Analytical continuation for in "s" from Schwartz reflection principle
    - It now makes sense to talk about dispersion relations for T and one can recover the physical region (in s-channel) as stated by Titchmarsh theorem
    - IMPORTANT: left branch cut does not correspond to physical region and we don't know what happens there --> fix it by re-expressing its contribution in terms of the right branch cut!

# 4. Narrowing down: pion scattering with Roy eqns.

- B. Ananthanarayan calls it "the theoretician's paradise" because ChiPT converges very well in it
- What is a pion?
    - Quark content
    - Pions are the Goldstone bosons of QCD at chiral limit
        - In chiral limit (with only 2 quarks) we have SU(2)xSU(2)xU(1)global symmetry (axial singlet has anomaly) --> we have 1 singlet vector current + 3 vector currents + 3 axial currents, all conserved, but 3 vector currents are spontaneously broken --> 3 Goldstone bosons, the pions (since 2 flavour isospin is a very good approx., pions are closer to being Goldstone bosons than other mesons which are Goldstones with 3 flavours --> this makes them much lighter than the rest of mesons and also makes SU(2) ChiPT converge much faster than SU(3) ChiPT)
        - If we introduced different masses for each quark flavour --> chiral symmetry explicitly broken
        - But if we introduce them with the same mass (very good approx for u, d) --> chiral symmetry is restored, but axial currents are no longer conserved (say that by conserved you mean Ward-Takahashi identity holds to 0)
        - Vector currents are still conserved but spontaneously broken --> we have massive "quasi-Goldstones" --> the pions. They do not acquire their mass from the Higgs mechanism, which is a spontaneous breaking of a gauge symmetry, but instead from the spontaneous breaking of a global symmetry!
- Changing basis: ideal for 2->2 scattering is to work in coupled isospin basis
- How do these new amplitudes relate to those of the uncoupled basis? --> Crossing matrices
- Link with 2: these relationships are true for every channel and in particular for well-defined isospin channels (quote thesis)

- Deriving Roy eqns. (don't do it all, just quote steps in thesis):
    - You start with eqn. A8
    - Re-express everything in terms of ds' --> using crossing matrix Csu
    - 2 subtractions ==> we need 2 external parameters --> evaluate T at selected points in s, t (and u, on-shell) --> threshold, since we can then relate it to scattering lengths, which are directly measurable --> the 2 parameters are scattering length of S wave in I=0,2 channels
    - We arrive at eqn. A.23
    - We now project in partial waves and get to the Roy eqns.

- More about the Roy eqns.
    - Anatomy of the Roy equations: ST + KT + DT
    - Anatomy of kernels --> cauchy in diagonals and logs
    - Only S and P waves have nonzero ST --> explain why with Re t^I_J expansion in momenta
    - Convergence / range of validity: Lehman ellipse
    - Non-unicity of solutions below unicity threshold
    - A practical implementation
        - Where does input come from? --> Parametrizations of Im t come from experiment (pion-nucleon) and we don't have much data for high energy ==> we don't know how the parametrization goes + many waves contribute
        - Solution: approximate these contributions as exchanges of Regge trajectories in isospin channels (explain Pomeron, w and 2w)


- Technical procedure:
    - Calculating kernels for S-G up to G-wave contrib.
    - Calculating subtraction terms
    - Calculating Regge from parametrizations
    - Calculating error bands

- Results:
    - Compare with literature (Jacobo, Rabán, Peláez, García)
    - Plot S, S+P, S+P+D, ..., until adding all to reconstruct T and study convergence --> was it worth it calculating up to G wave for my desired tolerance?

# 5. Conclusions

- 