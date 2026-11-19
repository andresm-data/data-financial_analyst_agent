# Database schema

DuckDB file `data/processed/finance.duckdb`, opened read-only. It has one table, `financials`: annual figures for 6 US banks taken from the SEC XBRL "company facts" API, using 10-K filings only (`form = 10-K`, `fp = FY`). There is one row per company, metric and fiscal year (660 rows).

**All amounts are in US dollars (USD), in whole dollars, not thousands or millions.** For example, `71372000000.0` is $71.372 billion.

## Table `financials`

Primary key: `(company, metric, fiscal_year)`. No column allows NULL.

| Column | Type | Meaning | Unit |
|---|---|---|---|
| `company` | VARCHAR | Bank ticker (see the list below). | none |
| `metric` | VARCHAR | Name of the figure (see the list below). | none |
| `fiscal_year` | INTEGER | Fiscal year, taken as the calendar year of `period_end`. It is not the year of the report that published the value. | year |
| `value` | DOUBLE | Amount of the metric. Negative values are losses (only in `NetIncomeLoss`). | USD |
| `unit` | VARCHAR | Always `'USD'`. | none |
| `period_end` | DATE | Last day of the fiscal period. For flow metrics (`Revenues`, `NetIncomeLoss`) the period is about 12 months; for balances (`Assets`, `Liabilities`, `StockholdersEquity`, `Deposits`) it is the balance date. | date |
| `filed_date` | DATE | Filing date of the 10-K this value comes from. If several 10-Ks reported the same year, it is the most recent one, so it is usually 1 to 4 years after `fiscal_year`. | date |

## Values of `company`

| `company` | Bank |
|---|---|
| `BAC` | Bank of America |
| `C` | Citigroup |
| `GS` | Goldman Sachs |
| `JPM` | JPMorgan Chase |
| `MS` | Morgan Stanley |
| `TFC` | Truist Financial Corp |

## Values of `metric`

There are exactly 6 metric names. Each is filled from US-GAAP XBRL tags, in order of preference, and the first tag with data for a year is used:

| `metric` | Meaning | Type | Fallback tags actually used |
|---|---|---|---|
| `Revenues` | Revenue for the year. | flow | GS 2011–2025 and MS 2015–2025: `RevenuesNetOfInterestExpense` (net revenues). TFC 2007–2025: `InterestAndDividendIncomeOperating` (interest and dividend income, not total revenue). |
| `NetIncomeLoss` | Net income; negative means a net loss. | flow | TFC 2007–2025: `ProfitLoss`, which includes noncontrolling interests. |
| `Assets` | Total assets at year end. | balance | none |
| `Liabilities` | Total liabilities at year end. | balance | none |
| `StockholdersEquity` | Total stockholders' equity at year end. | balance | C 2006–2007, JPM 2007 and TFC 2006–2023: `StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest` (includes noncontrolling interests). |
| `Deposits` | Total deposits at year end. | balance | none |

Because of the fallbacks, `Revenues` is not directly comparable across banks: for TFC it is interest and dividend income only.

## Years available

- `fiscal_year` goes from 2006 to 2025. Every bank has all 6 metrics for 2011–2025; earlier years are partial.
- First year per bank: BAC, C and TFC 2006; GS, JPM and MS 2007.
- Missing from 2008 on: GS `Deposits`, `Liabilities` and `StockholdersEquity` for 2008, and GS `Revenues` for 2008–2010.
- `period_end` is December 31 except for 6 rows with November 30: GS `NetIncomeLoss` 2007–2008, and MS `Revenues` and `NetIncomeLoss` 2007–2008. When an answer uses one of these rows, say that the fiscal year ended on November 30, not December 31.

## Known data issues

- Two rows have `value = 0`: BAC `Assets` 2008 and C `StockholdersEquity` 2006. Do not report them as real figures; say the value is not available.
