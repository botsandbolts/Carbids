# From CarBids auction prices to Pricing retail-sale estimates

## Three quantities, three labels

1. `asking_price`: an advertised amount (Pricing's original Craigslist target).
2. `auction_sale_price`: a published sold auction outcome (CarBids' target).
3. `retail_sale_price`: an actual/published retail transaction outcome for the target population. It requires its own labels.

Do not relabel quantity 1 or 2 as quantity 3. The datasets currently available here are 2015 wholesale auction sales and 2021 asking prices: source, time, condition, seller type and target meaning are confounded. No statistical sophistication can identify a contemporary retail transaction price from these data alone without additional assumptions or bridging observations. Neither dataset proves payment settlement.

## First deployable connection: external auction features

For a Pricing vehicle, calculate a frozen CarBids prediction `a(x)`, `log1p(a(x))`, its interval width, known-source/year flags as auxiliary features to the Pricing learner. The asking-to-auction gap may be used only in a retail-sale calibration task where the pre-sale ask is genuinely available at inference; it is excluded by default because it leaks the label when the downstream target is asking price. An asking-price learner remains an asking-price learner. These features are not guaranteed lower bounds or dealer acquisition costs.

`carbids.bridge` implements export plus a conservative training-VIN overlap index. It refuses VINs possibly seen during CarBids fitting and refuses missing VINs rather than assuming independence. A Bloom filter has false positives, so some independent vehicles may be conservatively excluded. For overlaps, refit CarBids excluding that VIN's group and regenerate held-out predictions. Feature schemas and every model must be versioned; a new upstream model changes downstream feature distribution.

```bash
python -m carbids.bridge index --cohort data/cohort.jsonl --model models/tabular
python -m carbids.bridge export --input pricing_vehicles.csv --output auction_features.csv
```

The index guards upstream training-vehicle leakage. It does not establish temporal validity, auction/retail overlap or current-market applicability. Reusing a 2015 auction feature on a 2021+ vehicle is unsupported extrapolation, even if VIN-disjoint. These fields are not silently injected into the original Pricing repo before a matched-cohort evaluation.

## Recommended calibrated model: hierarchical residual transport

Collect a bridge set with the same VIN's pre-sale asking price, eventual sale outcome, listing/sale dates, channel, seller type, condition and mileage. Keep unsold/censored listings too when modeling sale probability. Ensure each pairing is the same vehicle transaction, not a different auction years later.

A robust Bayesian residual model can start with

\[
\log(1+S_i)=\log(1+\hat A_i^{(-g)})+
\beta_0+h(X_i)+u_{make(i)}+v_{model(i)}+r_{channel(i)}+f(t_i)+\epsilon_i,
\quad \epsilon_i\sim t_\nu(0,\sigma_{channel(i)}).
\]

Here S is the target-domain sale price. The offset is an out-of-group Pricing ask estimate, not necessarily the seller's observed asking price. When the observed pre-sale ask is available at inference, it can be an additional feature, but the estimation task and evaluation must say so. A CarBids auction estimate may also enter `h(X)` as an auxiliary feature; do not force its coefficient to one or assume that the auction/retail gap is constant.

Use the posterior predictive median for the displayed point price under this log-Student-t formulation. Exponentiating a mean log price does not give the mean dollar price; an unbounded Student-t log residual has no finite exponential moment. If expected dollars are required, use a finite-moment positive distribution, an explicitly truncated predictive model, or a justified smearing estimator with tail diagnostics instead of silently exponentiating the fitted mean.

Use partial pooling for make/model/channel interactions: thin car segments shrink toward their parents while plentiful segments can differ. Student-t residuals tolerate unusually expensive examples without deleting half the market. A spline or random-walk time effect captures observed market changes; extrapolation uncertainty must grow outside the observed period. Fit time/channel effects only where the data support them: old wholesale and new retail data with no overlap cannot separately identify inflation and channel premium. Report sensitivity to priors rather than presenting prior-driven differences as learned facts.

For multimodal or strongly heterogeneous residuals, compare this with a mixture of experts by channel/vehicle segment or quantile LightGBM on the same auxiliary signals. Select using future, VIN-disjoint target-domain sales, not a source-domain score.

## Covariate shift: overlap before weighting

Fit a cross-fitted domain classifier q(x)=P(target domain | x), using pre-sale covariates only. Under the sampling class priors,

\[
w(x)=\frac{q(x)}{1-q(x)}\frac{n_{source}}{n_{target}}.
\]

This estimates a density ratio, not an asking-to-sale discount. Trim/clip extreme weights with reported sensitivity checks and report effective sample size `(sum w)^2 / sum(w^2)`. Inspect balance, domain-classifier calibration and common support. Unsupported vehicles should trigger abstention or an explicitly source-domain estimate, not falsely precise retail conversion. Make/model cardinality, condition scales and mileage units need harmonization before computing overlap.

A doubly robust transported **risk** estimate can combine an outcome-loss regression m(x) and these weights:

\[
\hat R_T=\frac1m\sum_{j\in T}\hat m(X_j)+
\frac1n\sum_{i\in S}\hat w(X_i)(L_i-\hat m(X_i)).
\]

Cross-fit both nuisance models. The result needs overlap and transport assumptions; it is not automatically a doubly robust individual price predictor. If `sale_price | X` differs by auction versus retail channel, covariate reweighting alone cannot fix it.

## Selection and unsold auctions

A sold-only dataset learns `E[price | X, sold]`, not the unconditional price or probability of sale. A separate sale-probability model needs sold **and** unsold auctions, reserves and exposure duration when available before prediction. Inverse-probability weighting requires selection-on-observables and positivity. Heckman selection requires defensible exclusion restrictions and distributional assumptions; it is not a magic correction for absent unsold records. Reserve-not-met highest bids are censored/inequality information, not completed-sale labels. Treat them in a separate censored likelihood only after validating the auction mechanism.

## Intervals: target-domain calibration, not borrowed certainty

Use an untouched target-domain calibration set of actual sale outcomes. Calibrate log absolute residuals or conformalized quantile-regression scores after model selection. Stratify by channel/segment only where enough data remain. Under covariate shift and valid density ratios, weighted conformal methods can transfer marginal coverage. They do not guarantee coverage when target mechanisms differ, selection is unmeasured, or time drift is unconstrained.

`weighted_conformal_radius` implements the weighted residual quantile with the test-point mass at infinity. An extreme target weight can return an infinite radius: the GUI should abstain. Merely normalizing weighted source residuals and calling their 80th percentile "80% target coverage" is incorrect. The current shipped baseline uses a validation residual band, not this untrained retail calibration.

## Evaluation and GUI contract

Use nested time-ordered, VIN-grouped fitting: upstream models/nuisances fitted only on allowed training data, calibration separate from selection, and a final untouched future retail-sales test. Compare ask-only, auction-only, stacked, hierarchical-calibrated and quantile variants on identical records. Report USD MAE/RMSE, log error, interval coverage/width and source/make/year/price-segment slices, plus effective sample sizes and unsupported-share. Use VIN/time-clustered bootstrap confidence intervals for paired error differences.

The GUI should show **asking estimate**, **auction-domain sale estimate**, and **calibrated retail-sale estimate** separately. The third remains unavailable until its model is fitted and validated. Show channel, sale-date coverage, calibration sample size, support diagnostics and fee basis. A humorous tooltip is welcome; a fabricated transaction target is not.

## Implementation status

Implemented: audited sale-only ingestion, real wholesale baseline and SHAP, MiniLM OOF training scaffold, upstream VIN-overlap rejection, auction-feature export, effective sample size and weighted conformal quantile primitive with tests.

Designed, not fitted: retail hierarchical residual model, domain classifier/transport weights, selection model, retail conformal calibration and Pricing retraining with the auxiliary features. Matched target-domain outcomes have not been collected in this project, so no calibrated retail-sale performance is claimed.

## Primary statistical references

- Tibshirani, Barber, Candès and Ramdas (2019), [Conformal Prediction Under Covariate Shift](https://proceedings.neurips.cc/paper/2019/hash/8fb21ee7a2207526da55a679f0332de2-Abstract.html).
- [Doubly robust calibration of prediction sets under covariate shift](https://academic.oup.com/jrsssb/article/86/4/943/7618755), JRSS B (2024).
- Heckman, [Sample Selection Bias As a Specification Error](https://www.nber.org/papers/w0172).

## Implemented structural transfer extension

The shared `marketbridge` package now fits age/mileage-constrained auction structure with empirical partial pooling, a lagged official BLS used-car CPI anchor, and a downstream asking-price experiment. It does not fit the matched-retail residual or selection models described above. See [INTEGRATION.md](INTEGRATION.md) for actual trained bundles, counterfactual decomposition, held-out results and explicit transport assumptions.
