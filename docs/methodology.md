# Methodology

Derivations behind the code in `src/mergerlab`. Notation: $n$ products, prices $p$, marginal
costs $c$, unit sales $q(p)$, Jacobian $J_{jk} = \partial q_j / \partial p_k$, absolute margins
$a = p - c$, Lerner margins $m = a / p$, ownership matrix $\Omega$ ($\Omega_{jk} = 1$ if
products $j$ and $k$ are priced by the same owner; fractional entries allowed).

## 1. Conventions

Three conventions are routinely confused, and each changes the answer. `mergerlab.units` wraps
every quantity in an object that records its convention and refuses to be used where another
is required.

**Shares.** Quantity shares $s_j = q_j / \sum_k q_k$ or revenue shares $w_j = p_j q_j / \sum_k p_k q_k$;
within the inside products (sum to one) or of a market that includes an outside good (sum to
$1 - s_0$). Converting between bases needs prices, and with an outside good also its price:

$$
w_j = \frac{p_j s_j}{\sum_k p_k s_k + p_0 s_0}.
$$

**Margins.** Lerner $m_j = (p_j - c_j)/p_j$ or absolute $a_j = p_j - c_j$.

**Diversion.** The quantity diversion ratio is

$$
D_{ij} = -\frac{\partial q_j / \partial p_i}{\partial q_i / \partial p_i},
$$

the value diversion is $D_{ij}\, p_j / p_i$, and the revenue diversion is
$-\frac{\partial r_j/\partial p_i}{\partial r_i/\partial p_i}$ with $r = pq$, which includes
the revenue effect of the price change of $i$ itself. With the own-price elasticity $e_{ii}$,

$$
D^{\text{rev}}_{ij} = D_{ij}\,\frac{p_j}{p_i}\,\frac{e_{ii}}{1 + e_{ii}}.
$$

AIDS-type calibrations, such as R's `antitrust`, report a third object, the share diversion
$S_{ij} = -\frac{\partial w_j/\partial p_i}{\partial w_i/\partial p_i} = -B_{ji}/B_{ii}$, a ratio of
budget-share slopes (R: `-t(slopes) / diag(slopes)`). Under the PCAIDS expenditure equation with
$k = \varepsilon_m + 1$, revenue shares $w$ and own elasticities $e_i$, differentiating
$q_j = Y_g w_j / p_j$ gives

$$
D_{ij} = \frac{p_i}{p_j}\,\frac{S_{ij}\,(e_i + 1 - k\,w_i) - k\,w_j}{e_i},
$$

which `Diversion.to_quantity` implements for the `SHARE` basis and `Diversion.to_share` inverts.
Only at a market elasticity of $-1$ ($k = 0$) does $S_{ij}$ equal the revenue diversion above; with
any other market elasticity, converting a share diversion as if it were a revenue diversion is
wrong. At $\varepsilon_m = -1.5$ in the example below the share diversion from A to B is still 0.375,
the quantity diversion is 0.160 (0.213 via the revenue conversion), and the CMCR is 11.5% and 8.5%
(16.4% and 12.6% via the revenue conversion); the R output at $\varepsilon_m = -0.7, -1.5, -2.5$ is
reproduced by the share conversion.

Quantity diversions supplied by a user must sum to at most one over the rivals of each product
(`Diversion.quantity` checks this). A demand system at unequal prices can imply quantity diversion
rows summing to more than one, because spending moves to cheaper goods, so diversions derived from
a demand system are built without that bound (`Diversion.from_demand`).

GUPPI, UPP and the CMCR are derived for the quantity diversion. Feeding the revenue diversion
into them applies the price ratio twice and the factor $e/(1+e)$ on top. In the
Epstein-Rubinfeld three-firm example (prices 2.9/3.4/2.2, revenue shares 20/30/50, own
elasticity $-3$, market elasticity $-1$, A and B merge) the revenue diversion from A to B is
0.375, the quantity diversion 0.213, and the CMCR of A is 16.7% with the quantity diversion and
32.9% with the revenue diversion.

## 2. Bertrand-Nash supply

Firm profits $\sum_k \Omega_{jk}(p_k - c_k) q_k$ give the first-order conditions

$$
F(p) = q(p) + \left(\Omega \circ J(p)^{\top}\right)(p - c) = 0 ,
$$

with $\circ$ the elementwise product. Pre-merger, observed prices identify marginal costs,
$c = p + (\Omega \circ J^{\top})^{-1} q$. Post-merger, $\Omega$ changes and the system is solved for $p$.

**Solver gate.** A candidate is accepted only if the scaled residual $\max_j |F_j(p)| / q_j$ is
below $10^{-10}$ and the prices lie where the demand system is valid. Otherwise
`EquilibriumNotFound` is raised with the residual reached from every start. The solver tries,
in order: the Morrow-Skerlos (2011) markup fixed point (logit-type demand), polished by damped
Newton steps with an exact complex-step Jacobian, then a hybrid Powell solver, over several
starting points. The textbook iteration $p \leftarrow c - (\Omega \circ J^{\top})^{-1} q$ is not a
contraction in general; see section 8.

**Zeta decomposition.** For logit-type demand the Jacobian splits as $J = \Gamma - \operatorname{diag}(\lambda)$
with $\lambda_j > 0$, and the first-order condition rearranges to

$$
p = c + \zeta(p), \qquad \zeta(p) = \frac{q + (\Omega \circ \Gamma^{\top})(p - c)}{\lambda}.
$$

## 3. Demand systems

### Logit

$s_j = \exp(\delta_j - \alpha p_j) / (1 + \sum_k \exp(\delta_k - \alpha p_k))$, so $\partial s_j/\partial p_j = -\alpha s_j(1 - s_j)$
and $\partial s_j/\partial p_k = \alpha s_j s_k$. With $m_k = p_k - c_k$ the first-order condition of product $j$ is

$$
1 - \alpha m_j + \alpha \sum_k \Omega_{jk} s_k m_k = 0
\quad\Longrightarrow\quad
m = \frac{1}{\alpha}\left(I - \Omega\,\operatorname{diag}(s)\right)^{-1}\mathbf 1 .
$$

For single-product firms $m_j = 1/(\alpha(1 - s_j))$; for a firm with several products every
product has the common markup $1/(\alpha(1 - s_f))$ with the firm share $s_f$. One margin identifies
$\alpha$; several margins give a least-squares fit for $1/\alpha$ and the residuals are reported.

*Logit ALM.* If the outside share is unknown, the inside shares $\tilde s$ are known and the total
shares are $b\tilde s$ with $b = 1 - s_0$. The markup condition then depends on $(\alpha, b)$ and two
margins that differ with firm share identify both. The fit is rejected, with an explanatory
`CalibrationError`, if $s_0$ is within $10^{-3}$ of 0 or 1 or if the margins do not vary with
firm share.

*Closed forms used as tests.* A single-product monopolist has the price
$p^{*} = c + \left(1 + W\!\left(e^{\delta - \alpha c - 1}\right)\right)/\alpha$ ($W$ the Lambert function).
Consumer surplus is the log-sum $CS = (M/\alpha)\ln\!\left(1 + \sum_j e^{\delta_j - \alpha p_j}\right)$, with $M$ the
market size.

### Nested logit

Berry (1994), parametrised as in Bjornerstedt and Verboven (2016), with the outside good as its own
nest. With $v_j = (\delta_j - \alpha p_j)/(1 - \sigma)$ and $D_g = \sum_{j \in g} e^{v_j}$:

$$
s_{j|g} = \frac{e^{v_j}}{D_g}, \qquad
s_g = \frac{D_g^{1-\sigma}}{1 + \sum_h D_h^{1-\sigma}}, \qquad s_j = s_{j|g}\, s_g .
$$

The Jacobian is $J = \Gamma - \operatorname{diag}(\lambda)$ with

$$
\lambda_j = \frac{M\alpha\, s_j}{1 - \sigma}, \qquad
\Gamma_{jk} = M\alpha\, s_j\left(\frac{\sigma}{1-\sigma}\,\mathbb 1[g(j)=g(k)]\, s_{k|g} + s_k\right).
$$

Given $\sigma$ and the shares the substitution matrix is proportional to $\alpha$, so $\alpha$ follows from
the margins exactly as for logit. Mean utilities are $\delta_j - \alpha p_j = \ln(s_j/s_0) - \sigma \ln s_{j|g}$.

### CES

Consumers with income $Y$ spend the revenue share $w_j = \beta_j p_j^{1-\sigma} / (1 + \sum_k \beta_k p_k^{1-\sigma})$ on
product $j$ and the rest on a numeraire outside good ($\sigma > 1$). Elasticities are
$\varepsilon_{jk} = (\sigma - 1) w_k - \sigma\,\mathbb 1[j = k]$, so a single-product firm has
$m_j = 1/(\sigma - (\sigma - 1) w_j)$. Every margin is decreasing in $\sigma$; $\sigma$ is calibrated
from the margins by (least squares) root finding. The utility is homothetic, so the compensating
variation is exact and closed form,

$$
CV = Y\left[\left(\frac{w_0(p')}{w_0(p)}\right)^{1/(\sigma - 1)} - 1\right].
$$

### Linear

$q = a + Bp$, with the diagonal of $B$ implied by margins and diversion ratios:
$B_{jj} = -q_j / \left[a_j - \sum_{k \ne j} \Omega_{jk} D_{jk} a_k\right]$ and $B_{kj} = -D_{jk} B_{jj}$ with absolute margins
$a$. Without supplied diversions they are proportional to quantity shares. The surplus is the
Marshallian line integral, exact only if $B$ is symmetric.

### PCAIDS

Epstein and Rubinfeld (2002). Within-market revenue shares are linear in log prices,
$w(p) = w^{0} + B \ln(p/p^{0})$ with $B = -k\,(\operatorname{diag}(w^0) - w^0 w^{0\top})$, which makes the
revenue diversion proportional to share, $w_j / (1 - w_i)$. Group expenditure responds to the
Divisia index with elasticity $\varepsilon_m + 1$ and elasticities are
$e_{ij} = B_{ij}/w_i + (\varepsilon_m + 1) w_j - \mathbb 1[i = j]$. One own elasticity $e$ of product $i$
fixes $B_{ii} = w_i\,(e + 1 - w_i(\varepsilon_m + 1))$ and therefore $k$; margins follow from the first-order
conditions and need not be supplied.

## 4. Concentration screens

$HHI = 10^4 \sum_f s_f^2$ for share fractions and $\Delta HHI = 2 s_A s_B \cdot 10^4$ for a merger
of A and B. Rule sets, checked against the official texts:

- **U.S. Merger Guidelines (2023), Guideline 1**: structural presumption if the post-merger HHI
  exceeds 1,800 and the increase exceeds 100, or if the merged share exceeds 30% and the increase
  exceeds 100. The presumption is rebuttable and not meeting it is no safe harbour.
- **U.S. Horizontal Merger Guidelines (2010), section 5.3**: unconcentrated below 1,500,
  moderately concentrated 1,500 to 2,500, highly concentrated above 2,500; an increase below
  100 ordinarily needs no further analysis; in moderately concentrated markets an increase of more than
  100 warrants scrutiny; in highly concentrated markets an increase of 100 to 200 warrants
  scrutiny and more than 200 is presumed likely to enhance market power.
- **EU Horizontal Merger Guidelines (2004)**, paragraphs 17 to 21: a combined share of 50% or more
  may itself indicate dominance (17); a share not above 25% indicates compatibility (18); post-merger HHI below 1,000
  (19); HHI of 1,000 to 2,000 with a delta below 250, or above 2,000 with a delta below 150 (20),
  except in special circumstances. Paragraph 21: none of this creates a presumption either way.

Screen labels follow the guidelines' wording. The EU screen never reports a presumption of harm:
a combined share of 50% or more is labelled "may indicate dominance" (paragraph 17). In the 2010
screen an increase of exactly 100 in a moderately concentrated market is neither "less than 100" nor
"more than 100"; it is labelled "concerns unlikely" and the summary says so.

The outside good is by default treated as atomistic (it adds nothing to the HHI); shares can
alternatively be renormalised to the inside products.

## 5. Unilateral-effects metrics

**GUPPI** (Farrell and Shapiro 2010; Moresi 2010). For two single-product firms,

$$
\text{GUPPI}_1 = D_{12}\, m_2\, \frac{p_2}{p_1},
$$

with quantity diversion and Lerner margin. In general $\text{GUPPI}_i = \sum_j (\Omega^{\text{post}} - \Omega^{\text{pre}})_{ij}\, D_{ij}\, m_j p_j / p_i$.
Net of a proportional cost saving $e_i$, $\text{UPP}_i/p_i = \text{GUPPI}_i - e_i(1 - m_i)$, which is zero at the
critical efficiency $e_i^{*} = \text{GUPPI}_i/(1 - m_i)$. (R's `antitrust::upp()` returns a different
quantity, a change in a weighted first-order condition, and is not used as an oracle.)

**CMCR** (Werden 1996). At fixed prices the first-order condition of $j$ is
$a_j - \sum_{k \ne j} \Omega_{jk} D_{jk} a_k = q_j / |J_{jj}|$, whose right side does not depend on ownership.
The margins that keep prices unchanged after the merger therefore solve

$$
\left(I - \Omega^{\text{post}} \circ D\right) a' = \left(I - \Omega^{\text{pre}} \circ D\right) a ,
\qquad \text{CMCR}_j = \frac{a'_j - a_j}{c_j}.
$$

For two single-product firms this reduces to Werden's closed form

$$
\text{CMCR}_1 = \frac{m_1 D_{12} D_{21} + m_2 D_{12}\,(p_2/p_1)}{(1 - m_1)(1 - D_{12} D_{21})}.
$$

Dollar synergies are $\sum_j \text{CMCR}^{\text{level}}_j\, q_j$ over the merging products.

**First-order approximation** (Jaffe and Weyl 2013). Write the post-merger first-order condition in
markup form, $h(p) = -\Delta^{-1}_{\text{pre}} F_{\text{post}}(p)$ with $\Delta_{\text{pre}} = \Omega^{\text{pre}} \circ J^{\top}$. At the
pre-merger prices $h(p^0) = g$ is the pricing pressure,
$g = -\Delta^{-1}_{\text{pre}}\left((\Omega^{\text{post}} - \Omega^{\text{pre}}) \circ J^{\top}\right)(p - c)$ (for two single-product
firms $g_1 = D_{12}(p_2 - c_2)$), and the price change is approximated by

$$
\Delta p \approx \rho\,\bigl(g + t\bigr), \qquad
t = \Delta^{-1}_{\text{pre}}\left(\Omega^{\text{post}} \circ J^{\top}\right)\Delta c, \qquad
\rho = -\left(\frac{\partial h}{\partial p}\right)^{-1}\Big|_{p^0},
$$

with $\Delta c$ the (negative) marginal-cost change from efficiencies. A cost change shifts the
post-merger first-order condition at $p^0$ by $-(\Omega^{\text{post}} \circ J^{\top})\Delta c$, which
in markup form adds $t$; for two single-product firms row $j$ of $t$ is $\Delta c_j - D_{jk}\Delta c_k$,
because a saving on one product also lowers the opportunity cost of the other. Adding $\Delta c$
itself instead of $t$ leaves a bias that is first order in $\Delta c$ (the cost effect is overstated by a factor of
1.75 in the symmetric Miller et al. logit example). The approximation is taken at the pre-merger prices, so
for a full merger it differs from the simulation by terms of the order of the merger itself; the tests
check that the ratio of the approximate to the simulated cost effect tends to one as the merger weakens.
`cost_pressure` returns $t$. The derivative is computed by complex-step
differentiation. For linear demand $h$ is linear and the approximation is exact. Koh (2025,
footnote 7) uses, for CES demand, the version in log prices with each condition divided by its price,
$\%\Delta p \approx M \cdot \text{GUPPI}$; `merger_pass_through(..., log_prices=True)` implements it.

**Hypothetical monopolist test.** The candidate market $S$ passes if the monopolist of $S$ (other
prices held fixed, or re-optimising) raises the price of some product by at least the SSNIP $t$ at
its profit-maximising prices (solver gate enforced). The critical-loss shortcut is $t/(t + m)$ with the
revenue-weighted Lerner margin $m$.

**Welfare.** Consumer harm is the compensating variation $CS(p^0) - CS(p^1)$: closed form for logit,
nested logit (inclusive value) and CES, a path integral for linear and PCAIDS. Producer surplus
is variable profit $(p - c)q$ at each demand form's own implied costs.

## 6. Uncertainty

Each Monte Carlo draw perturbs the supplied Lerner margins uniformly within $\pm h$ (clipped to 2% to 95%),
draws the nesting parameter and the PCAIDS market elasticity from stated ranges, and optionally draws
the linear model's diversion ratios row by row from a Dirichlet distribution centred on the
share-proportional default. Every demand form is recalibrated on the same perturbed margins, the post-merger
equilibrium is solved under the solver gate, and the merging firms' revenue-weighted average price change is
recorded. Draws that cannot be calibrated, or whose equilibrium fails the gate, are counted by cause,
reported and excluded from the quantiles. The ensemble quantiles weight each demand form equally
over its successful draws.

The bands are therefore conditional on inputs that can be calibrated: draws whose implied margins
leave $(0, 1)$ are inconsistent with the data, and excluding them truncates the prior to the feasible
set. For nested logit in the example the failures concentrate at high nesting parameters (0%, 10%,
34% and 65% across the four quarters of the range 0.1 to 0.7), so the effective prior is tilted
towards low values; `MonteCarloResult.covered_range` reports the range the successful draws cover.
The bands reflect the assumed ranges, not a posterior; they are a sensitivity analysis.

## 7. Concentration-based consumer harm (Koh 2025)

With no synergies the first-order change in consumer surplus is $\Delta CS = -\Delta p^{\top} q$ with $\Delta p = M g$
over the merging products. For logit and CES demand this factorises:

$$
\Delta CS = -V_0\,\rho_1\,\rho_2\,\Delta HHI, \qquad
\rho_1 = \frac{\phi}{(\phi - s_A)(\phi - s_B)},
$$

with $V_0 = N/\alpha$ and $\phi = 1$ for logit, $V_0 = Y/(\sigma - 1)$ and $\phi = \sigma/(\sigma - 1)$ for CES,
$\Delta HHI = 2 s_A s_B$ on the unit scale, and shares of the total market. For single-product
firms at unit prices

$$
\rho_2 = \frac{1}{2\phi}\left(M_{AA} + M_{BB} + M_{AB}\frac{s_A}{s_B} + M_{BA}\frac{s_B}{s_A}\right),
$$

which tends to one as the shares tend to zero. Derivation for logit: $g_A = s_B/(\alpha(1 - s_A)(1 - s_B))$ and
symmetrically for $B$; inserting into $-q^{\top} M g$ with $q = N s$ gives the factorisation above.
The formula covers the merging products with the rivals' prices held fixed. `harm_comparison`
therefore separates three numbers: the first-order harm, the exact compensating variation of the same
price change (merging products re-priced optimally, rivals fixed) and the full simulation. For
Heinz / Beech-Nut the first is within 6% of the second at $\sigma$ from 1.5 to 3, and the rivals'
price response (Gerber, +3.0% to +3.4%) accounts for 32% to 52% of the full harm.
`koh_decomposition` defines $\rho_2$ numerically from the first-order harm, so it holds at any prices, and
the tests assert that it equals the closed-form expression at unit prices.

## 8. Why the solver gate exists

For nested logit the iteration $p \leftarrow c - (\Omega \circ J^{\top})^{-1} q$ can cycle. In the
$\sigma = 0.6$ scenario of the pyblp comparison it alternates between two price vectors with largest
increases of +148.9% and +187.1% and scaled residuals of 6.6 and 11, while the equilibrium (pyblp, and
this package) has +43.5%. A loop that stops after a fixed number of iterations returns one of the two
points. The Morrow-Skerlos iteration from the same start has a scaled residual below $10^{-14}$ after 40
iterations (6.7e-15 in the committed trace). The 33% failure rate of the textbook iteration on random
nested-logit mergers refers to the undamped iteration with 10,000 steps and the sampling design in
`scripts/exp_solver_gate.py` (price coefficient U(1, 3), nesting parameter U(0, 0.9), three random
nests, mean utilities U(0.5, 2.5), costs U(0.4, 1.0)); a damped iteration with step 0.5 fails in 5 of
the 400 markets.

## References

- Berry, S. (1994). Estimating discrete-choice models of product differentiation. *RAND Journal of Economics* 25(2), 242-262.
- Bjornerstedt, J. and Verboven, F. (2016). Does merger simulation work? Evidence from the Swedish analgesics market. *American Economic Journal: Applied Economics* 8(3), 125-164.
- Epstein, R. and Rubinfeld, D. (2002). Merger simulation: a simplified approach with new applications. *Antitrust Law Journal* 69(3), 883-919.
- Farrell, J. and Shapiro, C. (2010). Antitrust evaluation of horizontal mergers: an economic alternative to market definition. *B.E. Journal of Theoretical Economics* 10(1).
- Jaffe, S. and Weyl, E. G. (2013). The first-order approach to merger analysis. *American Economic Journal: Microeconomics* 5(4), 188-218.
- Koh, P. S. (2025). Concentration-based inference for evaluating horizontal mergers. arXiv:2407.12924 (v5).
- Miller, N., Remer, M., Ryan, C. and Sheu, G. (2017). Upward pricing pressure as a predictor of merger price effects. *International Journal of Industrial Organization* 52, 216-247.
- Moresi, S. (2010). The use of upward price pressure indices in merger analysis. *Antitrust Source*, February 2010.
- Morrow, W. R. and Skerlos, S. J. (2011). Fixed-point approaches to computing Bertrand-Nash equilibrium prices under mixed-logit demand. *Operations Research* 59(2), 328-345.
- Werden, G. (1996). A robust test for consumer welfare enhancing mergers among sellers of differentiated products. *Journal of Industrial Economics* 44(4), 409-413.
- U.S. Department of Justice and Federal Trade Commission (2023). *Merger Guidelines*.
- U.S. Department of Justice and Federal Trade Commission (2010). *Horizontal Merger Guidelines*.
- European Commission (2004). *Guidelines on the assessment of horizontal mergers*, OJ C 31, 5.2.2004, p. 5.
