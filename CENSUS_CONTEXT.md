# Census regional context, 2016–2022

The US Census Bureau ACS five-year estimate **vintages** 2016–2022 span periods 2012–2016 through 2018–2022. They are pooled period estimates, not seven independent annual surveys. Adjacent vintages share most underlying years.

## Features

| Signal | Official detailed-table variable(s) | Interpretation |
|---|---|---|
| Household median income | B19013_001E, margin B19013_001M | Regional income in the vintage's inflation-adjusted dollars; not current individual buyer income |
| Population | B01003_001E | Regional scale; not population density without land area |
| Unemployment rate | B23025_005E / B23025_003E | Unemployed / civilian labor force |
| No-vehicle household share | B08201_002E / B08201_001E | Households reporting no available vehicle |
| Homeownership share | B25003_002E / B25003_001E | Owner-occupied / occupied housing units |
| Work-from-home share | B08301_021E / B08301_001E | Workers reporting working from home |

Estimate and margin-of-error fields are retained. Negative Census sentinel codes become missing. Invalid ratios and zero denominators become missing. A 90% MOE may be converted to an approximate standard error by dividing by 1.645 under the relevant normal approximation. Derived-ratio uncertainty also depends on numerator/denominator covariance; simply dividing their MOEs is not a valid ratio MOE.

## Release-aware joins

| Vintage | Public detailed-table release | Official reference |
|---|---|---|
| 2016 | 2017-12-07 | [2016 release](https://www.census.gov/programs-surveys/acs/news/data-releases/2016/release.html) |
| 2017 | 2018-12-06 | [2017 developer release](https://www.census.gov/data/developers/data-sets/acs-5year/2017.html) |
| 2018 | 2019-12-19 | [2019 updates](https://www.census.gov/programs-surveys/acs/news/updates/2019.html) |
| 2019 | 2020-12-10 | [Release announcement](https://www.census.gov/newsroom/press-releases/2020/acs-5-year.html) |
| 2020 | 2022-03-17 | [Delayed release announcement](https://www.census.gov/newsroom/press-releases/2022/acs-5-year-estimates.html) |
| 2021 | 2022-12-08 | [2021 release](https://www.census.gov/programs-surveys/acs/news/data-releases/2021/release.html) |
| 2022 | 2023-12-07 | [2022 developer release](https://www.census.gov/data/developers/data-sets/acs-5year/2022.html) |

The code starts availability on the following UTC day to avoid assuming a release time for a same-day sale, then selects only an available matching geography. No future-vintage filling is performed. Unknown/unsupported geography remains missing, with an explicit availability flag.

Current API values can incorporate later corrections. A release-aware calendar alone is **not proof of original point-in-time values**. Strict backtesting requires archived original releases or revision-time tracking. The cache records the actual retrieved data; obtain immutable snapshots before publishing historical performance claims about Census uplift.

## Statistical use in the Pricing bridge

Use regional features as contextual modifiers in a partially pooled sale/asking-gap model, with geography and time effects, not as deterministic discounts or explanations of an individual buyer. The effect of an area's income can vary by segment and sales channel. Pool sparse county effects toward state effects; regularize interactions. ACS measurement error can be propagated through a measurement-error model or sensitivity analysis using MOEs. Do not treat highly overlapping vintages as independent evidence of rapid market movement.

The auction yard's location may differ from buyer location. State context is coarse, while richer auction exports may support county context using an explicitly sourced FIPS. Postal ZIPs and Census ZCTAs are distinct and require an appropriate dated crosswalk; no naive ZIP-to-county join is supplied.

Census does not itself provide a used-car price inflation index. Updating 2015 vehicle prices to 2022 cannot be justified by household income growth alone. Use overlapping vehicle sales and, if desired, a separately sourced relevant price index, with prospective validation. No census variable fixes unobserved auction/retail selection or the absence of matched retail sale labels.

## Implementation and current status

Official fetcher, caching, state/county schemas, conservative as-of join, selected LightGBM feature support, inference date/panel requirements, GUI release calendar, CLI and tests are implemented. Regional values have **not been downloaded** because the API returned a Missing Key response. Configure `CENSUS_API_KEY` privately, or provide official saved data responses; the app substitutes no invented values.

The shipped 2015 wholesale model has no Census columns. Refit and compare an enriched model only on later sales for which appropriate data were available. Report matched-sample out-of-time MAE/RMSE, source/county/segment slices, interval coverage, missing-context share and sensitivity to ACS measurement error and vintage revisions.
