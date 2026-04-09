# 1. Introduction

- Objective: I want to study the Physics of pions. Characterizing their dynamics is so very important to understand the dynamics of other processes such as heavy meson decays or any interaction involving well defined 0, 1, 2 isospin channels
- Problem: we cannot procede with perturbative QCD because at low energies, QCD turns non-perturbative
    - Many ways to deal with this but I shall be using dispersive methods
    - Talk about the axiomatic approach to Physics: imposing a set of axioms which we believe to hold true, we can build a theory which is self-consistent and must yield the correct results if said axioms do indeed hold

# 2. An introduction to hadrons

- Objective of this section: to give an understanding of the actual physical object whose dynamics we shall be studying

- Talk about QCD a bit phenomenologically
- Talk about quarks, how they are confined into hadrons, about the light meson octet and about pions
- A deeper dive into pions: isospin limit as motivated by Heisenberg --> isospin limit and isospin breaking --> maybe mention they are the Goldstone bosons associated to the spontaneous breaking of the chiral symmetry in 2 flavours + Adler's theorem for suppressed interaction
- At low energies (below hadronic scale), QCD is non-perturbative in its coupling constant and therefore one cannot treat the study of pions at low energies with this tool --> how do we proceed? --> Dispersive methods

# 3. An intro to dispersive methods

- Objective of this section: to derive the dispersion relations for $T$ of 2->2 scattering process and explain the theory underlying them
- Preview: give intuition of what a dispersion relations actually does --> make analogy with Kramers-Kronig (the idea is the same althopugh the interpretation is different, since it is now a probability amplitude)
- Talk about invariant amplitude and connection to S matrix --> instead of obtaining it with Feynman diagrams, we calculate it exactly and in a model-independent way (this was developed in the 60s, before we had a theory for QCD)
- Beginning of explanation of dispersion relations:
    - We impose:
          - Unitarity: if we begin with a given asimptotic state, we must end in another one --> $\displaystyle\sum_\beta |\langle \alpha |S |\beta\rangle|^2=\displaystyle\sum_\beta |\langle \alpha |S |\beta\rangle\langle \beta S^\dagger |\alpha\rangle = \langle \alpha SS^\dagger |\alpha \rangle$ and this must be equal to the normalization --> explore the consequences of unitarity --> optical theorem and imaginary part gives scattering amplitude
          - Lorentz invariance: Einstein's principle of relativity: (in the absence of gravity) the laws of Physics are the same for all inertial frames of reference --> all (intertial) observers must agree on what the probability amplitude of a given process is: $\langle \alpha|S|\beta\rangle$ must be a scalar. We choose the Lorentz-invariant normalization and demand $S$ to be a Lorentz scalar so that $T'=T$
          - Analyticity
          - Causality --> strongly linked through Hilbert transform
    - Introduce Mandesltam variables and Mandelstam triangle
    - Promote "s" to complex variable --> explain how different sections of the complex plane connect to real Physics in different channels (this is crossing, enabled by time reversal invariance)
    - Explain analytical structure through Mandelstam hypothesis

# 4. Narrowing down: pion scattering with Roy eqns.

- B. Ananthanarayan calls it "the theoretician's paradise" because ChiPT converges very well in it
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