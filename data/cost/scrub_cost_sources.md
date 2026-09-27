# Weather scrub cost research — Cape Canaveral / Kennedy Space Center

Research date: 2026-09-27. Scope: Falcon 9 (Cape Canaveral SFS / KSC), with Shuttle/SLS context
where relevant. Every figure below is tagged with its source and date; figures with no primary
source are explicitly marked as such.

## Bottom line

| | LOW | CENTRAL | HIGH |
|---|---:|---:|---:|
| **Direct cost per weather scrub** | $100,000 | $400,000 | $1,050,000 |
| **Delay/opportunity cost, per day** | $20,000 | $267,000 | $1,000,000 |

No agency or company publishes an actual "cost of one Falcon 9 weather scrub." The numbers above
are built by anchoring to the few real dollar figures that do exist and stating, explicitly, the
assumption used to turn each one into a per-scrub or per-day estimate. See `scrub_cost_model.json`
for the machine-readable version and per-component arithmetic.

## 1. Direct scrub cost, Falcon 9

### Propellant / pressurant

- **Elon Musk, ~2012 interview:** Falcon 9's fuel, oxidizer, and pressurant amount to roughly
  **$200,000**, about 0.3% of the ~$60M mission cost, for the *whole vehicle* (not per stage).
  [NextBigFuture transcript](https://www.nextbigfuture.com/2012/05/interview-elon-musk-of-spacex-talks.html)
  (page dated 2017-04-07, reproducing a 2012 interview).
- **Musk, reported 2017:** refueling a reusable first stage for another flight costs about
  **$200,000-$300,000**. [Space.com, 2017-04-10](https://www.space.com/36412-spacex-completely-reusable-rocket-elon-musk.html)
  — *(note: this is a first-stage refuel estimate, not the same accounting as the 2012 full-vehicle figure)*.
- **No SpaceX or third-party source quantifies helium or nitrogen cost per launch**, and none
  states what fraction of a scrub's propellant load is actually vented/lost vs. detanked and
  reused. An informal 2014 Reddit/r/spacex thread discusses the detank-and-recycle operation
  qualitatively but its dollar figures are explicitly speculative
  ([source](https://www.reddit.com/r/spacex/comments/28tgm6/how_much_does_a_scrub_cost/), 2014-06-22).
- A third-party 2023 blog estimate of raw LOX+RP-1 commodity cost (~$512,000) exists but conflicts
  sharply with Musk's own figure and is not an accounting number
  ([Space Insider, 2023-06-13](https://spaceinsider.tech/2023/06/13/how-much-does-rocket-fuel-cost/)).
- **Verdict:** the defensible number is Musk's own $200,000-$300,000 estimate for propellant/
  pressurant/refueling. What fraction of that is actually *lost* in a scrub (as opposed to
  recovered) is unsourced — modeled here as an assumption (see JSON).

### Range / Space Force support

- No public per-launch price exists for Eastern Range / Space Launch Delta 45 (SLD 45) support.
  SLD 45 obligates **roughly $300 million/year** in contracts, per a September 2021 small-business
  briefing ([USSF trifold, Sept. 2021](https://www.airforcesmallbiz.af.mil/Portals/58/Brochures/USSF%20Trifold%20Sept%2021.pdf))
  — an organization-wide figure, not a per-launch or per-scrub charge.
- As of June 2025, the Space Force moved to a task-order model where commercial launch companies
  pay directly for range services/upgrades ([Defense News, 2025-06-04](https://www.defensenews.com/space/2025/06/04/space-force-shifts-upfront-range-upgrade-costs-to-commercial-firms/);
  [Space Systems Command](https://www.ssc.spaceforce.mil/Newsroom/Article-Display/Article/4205247/space-systems-command-award-advances-eastern-and-western-range-capability)),
  but no standard price card or representative Falcon 9 task-order amount is public.
- **Verdict:** modeled here only as an assumption — the $300M/yr figure divided across ~109
  Cape launches/year (2025 rate, see cadence below) implies ~$2.75M of average range-support
  spend per launch; the model assumes a scrub consumes a small fraction of that as marginal
  labor/overtime. This is an allocation, not a marginal cost — flagged as such.

### SpaceX "standing army" (launch team) and recovery fleet (droneships, GO ships)

- **No public source** gives SpaceX's daily launch-team labor cost or a droneship/recovery-vessel
  day rate. SpaceX is private and does not publish mission-level cost accounting. A Starlink case
  study gives a connectivity cost for the recovery fleet (~$5,000/vessel/month for Starlink
  internet, replacing >$165,000/month of prior satellite bandwidth —
  [SpaceX/Navy Starlink case study PDF](https://www.starlink.com/public-files/SpaceXNavyStarlink-case-study.pdf))
  but that is bandwidth cost, not vessel operating cost.
- **Verdict:** omitted from the model rather than invented. If your app needs a placeholder,
  treat it as unquantified rather than presenting a number as sourced.

### Shuttle-era and SLS reference points (context only, not used in the Falcon 9 model)

- The commonly repeated claim "a Shuttle scrub cost about $1 million" **could not be traced to a
  primary NASA or GAO document** in this research pass.
- The closest sourced $1M figure found is for a **landing diversion** (not a launch scrub): flying
  the orbiter back from California on the Shuttle Carrier Aircraft cost "approximately $1 million
  in fuel... plus overtime and travel expenses for the turnaround team"
  ([archived AmericaSpace/Chien article, 2002-12-06](https://ictnews.org/archive/herringtons-shuttle-sets-new-record-for-landing-delays/)).
  This is a different event (post-mission landing logistics) from a pre-launch weather scrub, so
  treat the popular "$1M per scrub" line as an unverified, commonly repeated estimate, not a
  primary-sourced Shuttle launch-scrub figure.
- Former Shuttle launch/program director **Wayne Hale** has publicly written that clean per-event
  Shuttle cost figures are contested and that simple "annual budget ÷ missions flown" math is
  "totally inaccurate"
  ([Wayne Hale's blog, 2019-11-09](https://waynehale.wordpress.com/2019/11/09/what-figure-did-you-have-in-mind/)).
- **SLS/Orion:** NASA OIG puts the *total production-and-operations cost of a single SLS/Orion
  launch* at **$4.1 billion** for Artemis I-IV
  ([IG-21-018, March 2021](https://oig.nasa.gov/wp-content/uploads/2024/02/ig-21-018.pdf?emrc=4d982c);
  reaffirmed in [CT-2022-01, Feb. 2022](https://oig.nasa.gov/wp-content/uploads/2024/02/ct-2022-01.pdf)).
  GAO's [GAO-23-105609 (Sept. 2023)](https://www.gao.gov/assets/gao-23-105609.pdf) covers SLS cost
  transparency generally. **None of these reports break out a scrub-specific dollar figure** — the
  $4.1B is total per-launch program cost, not an incremental scrub cost, and is not used in the
  per-scrub model.

## 2. Opportunity / delay cost

- **Falcon 9 list price:** raised from $62M to **$67M** in March 2022, per SpaceX's Capabilities
  & Services page ([Space.com, 2022-03-23](https://www.space.com/spacex-raises-prices-launch-starlink-inflation)).
  Later secondary sources (industry roundups, not primary) cite figures in the $70-74M range for
  2024-2026, but these are not confirmed against a current SpaceX-published price sheet in this
  pass. This is provided as context on launch value, not used directly as a delay-cost input.
- **GEO satellite revenue lost per day of delay:** no directly published figure was found. The
  closest sourced analog is Maxar's 2022 settlement with EchoStar/Hughes over the delayed
  Jupiter-3/EchoStar-24 GEO broadband satellite: Maxar agreed to pay EchoStar **~$8 million/month**
  (~$267,000/day) in compensation for the delay
  ([Kratos Constellations / Space Intel Report, 2022-11-23](https://www.kratosspace.com/constellations/articles/maxar-concessions-to-echostar-for-jupiter-3-delay-show-limit-of-force-majeure)).
  This is a satellite-*manufacturing*-delay compensation figure, not a launch-scrub-specific
  revenue-loss number, but it is the best available sourced order-of-magnitude proxy for "value of
  one day" of a large GEO satellite. Used as the CENTRAL value in the model; LOW and HIGH bounds
  around it are unsourced assumptions (see JSON `source` field for exact reasoning).

## 3. Space Coast launch cadence and FAA forecast

| Year | Cape Canaveral SFS + KSC launches | Source |
|---|---:|---|
| 2023 | 72 | [Florida Today via USA Today Network, 2025-07-17](https://eu.app.com/story/tech/science/space/2025/07/17/will-cape-canaveral-see-unprecedented-100-rocket-launches-during-2025-in-brevard-county-florida/85196064007/) |
| 2024 | 93 (record at the time) | same source |
| 2025 | 109 (final; 101 SpaceX, 6 ULA, 2 Blue Origin, 4 crewed) | [Central Florida Public Media, 2025-12-26](https://www.cfpublic.org/space/2025-12-26/florida-rocket-launches-break-record-reaching-triple-digits-in-2025); corroborated by [The Space Review, 2026-02-09](https://thespacereview.com/article/5156/1) |
| 2026 (forecast) | "somewhere between 100 and 120" | [45th/SLD45 Commander Brian Chatman, via Spectrum News 13, 2025-12-29](https://mynews13.com/fl/orlando/news/2025/12/29/record-number-of-space-coast-launches-expected-in-2026) |
| 2035 (forecast) | up to ~300-500/year off the Space Coast | Chatman, cited in [The Space Review, 2026-02-09](https://thespacereview.com/article/5156/1) and [Spectrum News 13, 2025-12-29](https://mynews13.com/fl/orlando/news/2025/12/29/record-number-of-space-coast-launches-expected-in-2026) |

2020-2022 Cape-specific annual counts were not confirmed against a citable Cape-specific source in
this research pass and are omitted rather than estimated (global orbital-launch-by-year data exists
but is not Cape-specific, so it was left out to avoid conflating the two).

**FAA licensing trend (national, not Cape-specific, but the standard citation for "growth in
licensed operations"):**
- FAA ended **FY2024 with a record 148 licensed commercial space operations, up >30% year over
  year**, and forecast the number "may more than double by FY 2028"
  ([FAA newsroom, 2024-11-14](https://www.faa.gov/newsroom/new-record-faa-licensed-commercial-space-operations-aerospace-rulemaking-committee)).
- FAA's **Aerospace Forecast FY2025-2045** projects, for licensed commercial space operations:
  high-case growth from **~183 (FY2025) to 566 (FY2034)**; low-case from **~174 (FY2025) to 259
  (FY2034)** ([FAA report PDF, 2025](https://www.faa.gov/data_research/aviation/aerospace_forecasts/2025-commercial-space.pdf);
  summarized in [New Space Economy, 2025-08-11](https://newspaceeconomy.ca/2025/08/11/faa-forecasts-indicate-substantial-expansion-of-commercial-space-activity-by-fy-2034/)).
- FAA reached its **1,000th licensed/permitted commercial space operation on 2025-08-14**
  ([FAA Commercial Space Transportation page](https://www.faa.gov/space)).

## Sources table

| # | Claim | Source | Date |
|---|---|---|---|
| 1 | Falcon 9 propellant/pressurant ~$200,000 (whole vehicle) | [NextBigFuture, Musk interview transcript](https://www.nextbigfuture.com/2012/05/interview-elon-musk-of-spacex-talks.html) | 2012 interview, page 2017-04-07 |
| 2 | First-stage refuel ~$200,000-$300,000 | [Space.com](https://www.space.com/36412-spacex-completely-reusable-rocket-elon-musk.html) | 2017-04-10 |
| 3 | Detank/recycle operational description (no cost data) | [Reddit r/spacex](https://www.reddit.com/r/spacex/comments/28tgm6/how_much_does_a_scrub_cost/) | 2014-06-22 |
| 4 | SLD45 ~$300M/yr contracting obligations | [USSF Small Business trifold](https://www.airforcesmallbiz.af.mil/Portals/58/Brochures/USSF%20Trifold%20Sept%2021.pdf) | Sept. 2021 |
| 5 | Space Force shifts range costs to commercial task orders | [Defense News](https://www.defensenews.com/space/2025/06/04/space-force-shifts-upfront-range-upgrade-costs-to-commercial-firms/) | 2025-06-04 |
| 6 | SLS/Orion ~$4.1B per launch (context only) | [NASA OIG IG-21-018](https://oig.nasa.gov/wp-content/uploads/2024/02/ig-21-018.pdf?emrc=4d982c) | March 2021 |
| 7 | SLS cost transparency | [GAO-23-105609](https://www.gao.gov/assets/gao-23-105609.pdf) | Sept. 2023 |
| 8 | Shuttle landing-diversion ~$1M fuel+overtime (not a launch scrub) | [Archived AmericaSpace/Chien article](https://ictnews.org/archive/herringtons-shuttle-sets-new-record-for-landing-delays/) | 2002-12-06 |
| 9 | Shuttle per-event cost figures are contested | [Wayne Hale's blog](https://waynehale.wordpress.com/2019/11/09/what-figure-did-you-have-in-mind/) | 2019-11-09 |
| 10 | Falcon 9 list price $62M→$67M | [Space.com](https://www.space.com/spacex-raises-prices-launch-starlink-inflation) | 2022-03-23 |
| 11 | Maxar→EchoStar ~$8M/month delay compensation (Jupiter-3) | [Kratos Constellations / Space Intel Report](https://www.kratosspace.com/constellations/articles/maxar-concessions-to-echostar-for-jupiter-3-delay-show-limit-of-force-majeure) | 2022-11-23 |
| 12 | Cape/KSC launches 2023 (72), 2024 (93) | [Florida Today / USA Today Network](https://eu.app.com/story/tech/science/space/2025/07/17/will-cape-canaveral-see-unprecedented-100-rocket-launches-during-2025-in-brevard-county-florida/85196064007/) | 2025-07-17 |
| 13 | Cape/KSC launches 2025 (109 final) | [Central Florida Public Media](https://www.cfpublic.org/space/2025-12-26/florida-rocket-launches-break-record-reaching-triple-digits-in-2025) | 2025-12-26 |
| 14 | 2025 total corroboration + 2035 outlook (~300-500/yr) | [The Space Review](https://thespacereview.com/article/5156/1) | 2026-02-09 |
| 15 | 2026 forecast (100-120 launches) | [Spectrum News 13](https://mynews13.com/fl/orlando/news/2025/12/29/record-number-of-space-coast-launches-expected-in-2026) | 2025-12-29 |
| 16 | FAA record 148 licensed ops FY2024, forecast to double by FY2028 | [FAA newsroom](https://www.faa.gov/newsroom/new-record-faa-licensed-commercial-space-operations-aerospace-rulemaking-committee) | 2024-11-14 |
| 17 | FAA Aerospace Forecast FY2025-2045 (183→566 high case) | [FAA PDF](https://www.faa.gov/data_research/aviation/aerospace_forecasts/2025-commercial-space.pdf); summarized in [New Space Economy](https://newspaceeconomy.ca/2025/08/11/faa-forecasts-indicate-substantial-expansion-of-commercial-space-activity-by-fy-2034/) | 2025 report / 2025-08-11 summary |
| 18 | FAA reaches 1,000th licensed operation | [FAA Commercial Space Transportation](https://www.faa.gov/space) | 2025-08-14 |

## Explicitly unresolved / not found

- No SpaceX, NASA, GAO, or Space Force document states a per-scrub dollar figure for Falcon 9.
- No public helium or nitrogen consumption cost per launch.
- No public SpaceX launch-team ("standing army") daily labor cost.
- No public droneship/recovery-vessel day rate.
- No public per-launch (as opposed to annual/organizational) Eastern Range price.
- No public GEO-satellite-revenue-per-day-of-launch-delay figure (the Maxar/EchoStar figure used
  is a manufacturing-delay analog, not a launch-scrub-specific number).
- No primary NASA/GAO source for the popular "~$1M per Shuttle scrub" claim.
- Cape-specific (as opposed to global) launch counts for 2020-2022.
