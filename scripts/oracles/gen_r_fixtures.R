# Reference values from the CRAN package 'antitrust' (Taragin and Sandfort; CC0).
#
# Writes tests/data/r_antitrust.json. The test suite reads the JSON and never calls R.
# Run from the repository root; the packages live in a project-local library:
#
#   mkdir -p .rlib
#   R_LIBS_USER="$PWD/.rlib" Rscript -e 'install.packages(c("antitrust", "jsonlite"), lib = Sys.getenv("R_LIBS_USER"), repos = "https://cloud.r-project.org")'
#   R_LIBS_USER="$PWD/.rlib" Rscript scripts/oracles/gen_r_fixtures.R
#
# Only exactly identified, converged examples are used.

suppressMessages({
  library(antitrust)
  library(jsonlite)
})

out <- list(
  generator = "scripts/oracles/gen_r_fixtures.R",
  r_version = R.version.string,
  antitrust_version = as.character(packageVersion("antitrust")),
  jsonlite_version = as.character(packageVersion("jsonlite"))
)
quiet <- function(expr) suppressWarnings(expr)

# 1. Epstein-Rubinfeld three-firm PCAIDS example ------------------------------------
shares <- c(0.2, 0.3, 0.5)
prices <- c(2.9, 3.4, 2.2)
owner_pre <- c("A", "B", "C")
owner_post <- c("A", "A", "C")
pc <- quiet(pcaids(shares, -3, -1, ownerPre = owner_pre, ownerPost = owner_post,
                   labels = owner_pre, priceStart = rep(0, 3)))
out$pcaids_epstein_rubinfeld <- list(
  revenue_shares = shares,
  prices = prices,
  own_elasticity_first = -3,
  market_elasticity = -1,
  price_change_pct = unname(calcPriceDelta(pc) * 100),
  elasticities = unname(elast(pc, TRUE)),
  margins = unname(calcMargins(pc, TRUE)),
  revenue_diversion = unname(diversion(pc, TRUE)),
  cmcr_pct = unname(cmcr(pc)),
  hhi_pre = sum((shares * 100)^2),
  hhi_post = sum((c(0.5, 0.5) * 100)^2)
)

# 2. Werden (1996) Table 1 grid via cmcr.bertrand ------------------------------------
margins <- seq(0.4, 0.7, 0.1)
divs <- seq(0.05, 0.25, 0.05)
grid <- matrix(NA_real_, nrow = length(divs), ncol = length(margins))
for (i in seq_along(divs)) {
  for (j in seq_along(margins)) {
    d <- matrix(c(-1, divs[i], divs[i], -1), 2)
    grid[i, j] <- unname(cmcr.bertrand(c(1, 1), c(margins[j], margins[j]), d, c(1, 0))[1])
  }
}
out$cmcr_werden_table1 <- list(margins = margins, diversions = divs, cmcr_pct = grid)

# 3. cmcr.bertrand documentation example (three products, one merging party owns one) ----
p3 <- c(50, 45, 70)
m3 <- c(0.3, 0.4, 0.6)
d3 <- matrix(c(-1, .5, .01, .6, -1, .1, .02, .2, -1), ncol = 3)
out$cmcr_bertrand_doc <- list(
  prices = p3, margins = m3, diversion = unname(d3),
  owner_pre = c(1, 0, 0), owner_post_all = TRUE,
  cmcr_pct = unname(cmcr.bertrand(p3, m3, d3, c(1, 0, 0)))
)

# 4. Logit with known outside share, one margin (exactly identified) ------------------
lp <- c(1.0, 1.3, 0.9, 1.1)
lq <- c(0.15, 0.25, 0.20, 0.10)
lm <- c(0.40, NA, NA, NA)
lo_pre <- c("A", "B", "C", "D")
lo_post <- c("A", "B", "A", "D")
lg <- quiet(logit(lp, lq, lm, ownerPre = lo_pre, ownerPost = lo_post, labels = lo_pre))
out$logit_known_outside <- list(
  prices = lp, quantity_shares = lq, margins = lm, owner_pre = lo_pre, owner_post = lo_post,
  alpha = unname(-lg@slopes$alpha), mean_utility = unname(lg@slopes$meanval),
  price_change_pct = unname(calcPriceDelta(lg) * 100),
  fitted_margins = unname(calcMargins(lg, TRUE)),
  compensating_variation = unname(CV(lg)),
  note = "R solves for alpha numerically (tolerance about 1e-6); compare accordingly"
)

# 5. Logit with a multi-product firm before the merger --------------------------------
mp_pre <- c("A", "A", "B", "C")
mp_post <- c("A", "A", "A", "C")
lg2 <- quiet(logit(lp, lq, c(NA, 0.35, NA, NA), ownerPre = mp_pre, ownerPost = mp_post,
                   labels = lo_pre))
out$logit_multiproduct <- list(
  prices = lp, quantity_shares = lq, margins = c(NA, 0.35, NA, NA),
  owner_pre = mp_pre, owner_post = mp_post,
  alpha = unname(-lg2@slopes$alpha),
  price_change_pct = unname(calcPriceDelta(lg2) * 100)
)

# 6. Logit ALM (outside share unknown) on exactly identified synthetic margins ---------
alpha_true <- 2.5
s_tot <- c(0.15, 0.25, 0.20)
s0_true <- 1 - sum(s_tot)
ap <- c(1.0, 1.3, 0.9)
a_own <- c("A", "B", "C")
abs_margin <- 1 / (alpha_true * (1 - s_tot))
am <- abs_margin / ap
alm <- quiet(logit.alm(ap, s_tot / sum(s_tot), c(am[1], am[2], NA), ownerPre = a_own,
                       ownerPost = c("A", "A", "C"), labels = a_own))
out$logit_alm_exact <- list(
  prices = ap, inside_shares = s_tot / sum(s_tot), margins = c(am[1], am[2], NA),
  owner_pre = a_own, owner_post = c("A", "A", "C"),
  alpha_true = alpha_true, outside_share_true = s0_true,
  alpha = unname(-alm@slopes$alpha), shareInside = unname(alm@shareInside),
  price_change_pct = unname(calcPriceDelta(alm) * 100)
)

# 7. Linear demand from margins and given quantity diversions (no symmetry) ------------
q <- c(15, 25, 20, 10) / lp
lmg <- c(0.4, 0.3, 0.35, 0.45)
D <- matrix(c(0, .30, .25, .15, .20, 0, .30, .20, .25, .35, 0, .10, .30, .20, .25, 0), 4, byrow = TRUE)
diag(D) <- -1
li <- quiet(linear(lp, q, lmg, diversions = D, symmetry = FALSE, ownerPre = lo_pre,
                   ownerPost = lo_post, labels = lo_pre))
out$linear_given_diversion <- list(
  prices = lp, quantities = q, margins = lmg, diversion = unname(D),
  owner_pre = lo_pre, owner_post = lo_post,
  price_change_pct = unname(calcPriceDelta(li) * 100),
  marginal_costs = unname(li@mcPre),
  slopes = unname(li@slopes)
)

# 8. PCAIDS away from market elasticity -1 --------------------------------------------
# With a market elasticity other than -1 group expenditure responds to prices, and R's AIDS
# diversion, -t(slopes) / diag(slopes), is a ratio of budget-share slopes, not of revenue changes.
pc_other <- list()
for (em in c(-1.5, -0.7, -2.5)) {
  pcm <- quiet(pcaids(shares, -3, em, ownerPre = owner_pre, ownerPost = owner_post,
                      labels = owner_pre, priceStart = rep(0, 3)))
  pc_other[[length(pc_other) + 1]] <- list(
    market_elasticity = em,
    price_change_pct = unname(calcPriceDelta(pcm) * 100),
    elasticities = unname(elast(pcm, TRUE)),
    margins = unname(calcMargins(pcm, TRUE)),
    share_diversion = unname(diversion(pcm, TRUE)),
    cmcr_pct = unname(cmcr(pcm))
  )
}
out$pcaids_market_elasticity <- list(
  revenue_shares = shares, prices = prices, own_elasticity_first = -3, cases = pc_other
)

dir.create("tests/data", showWarnings = FALSE, recursive = TRUE)
write_json(out, "tests/data/r_antitrust.json", digits = NA, auto_unbox = TRUE, pretty = TRUE,
           na = "null")
cat("wrote tests/data/r_antitrust.json\n")
