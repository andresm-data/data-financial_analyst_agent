-- Expected answers for the figure questions in eval/questions.csv (ids 26-40).
-- Source: table `financials` in data/processed/finance.duckdb (open read-only).
-- All values are in USD. Conventions:
--   * Amounts are reported in USD billions, rounded to 1 decimal.
--   * Growth is reported as percent (pct) or absolute change (USD billions),
--     rounded to 1 decimal, as stated in each query.
--   * Rounding happens only in the final SELECT, never on intermediate values.

-- id: 26
-- Which bank had the highest total assets in fiscal year 2025?
SELECT company, ROUND(value / 1e9, 1) AS assets_usd_bn
FROM financials
WHERE metric = 'Assets' AND fiscal_year = 2025
ORDER BY value DESC
LIMIT 1;

-- id: 27
-- Which bank had the lowest net income in fiscal year 2025?
SELECT company, ROUND(value / 1e9, 1) AS net_income_usd_bn
FROM financials
WHERE metric = 'NetIncomeLoss' AND fiscal_year = 2025
ORDER BY value ASC
LIMIT 1;

-- id: 28
-- Which bank had the largest total deposits in fiscal year 2020?
SELECT company, ROUND(value / 1e9, 1) AS deposits_usd_bn
FROM financials
WHERE metric = 'Deposits' AND fiscal_year = 2020
ORDER BY value DESC
LIMIT 1;

-- id: 29
-- Which bank had the lowest stockholders' equity in fiscal year 2025?
SELECT company, ROUND(value / 1e9, 1) AS equity_usd_bn
FROM financials
WHERE metric = 'StockholdersEquity' AND fiscal_year = 2025
ORDER BY value ASC
LIMIT 1;

-- id: 30
-- How much larger were JPMorgan Chase's total assets than Bank of America's in fiscal year 2025?
SELECT
    ROUND(jpm.value / 1e9, 1)               AS jpm_assets_usd_bn,
    ROUND(bac.value / 1e9, 1)               AS bac_assets_usd_bn,
    ROUND((jpm.value - bac.value) / 1e9, 1) AS difference_usd_bn
FROM financials AS jpm
JOIN financials AS bac
  ON bac.metric = jpm.metric AND bac.fiscal_year = jpm.fiscal_year
WHERE jpm.company = 'JPM' AND bac.company = 'BAC'
  AND jpm.metric = 'Assets' AND jpm.fiscal_year = 2025;

-- id: 31
-- Which reported higher revenues in fiscal year 2024, Goldman Sachs or Morgan Stanley, and by how much?
SELECT
    ROUND(gs.value / 1e9, 1)                    AS gs_revenues_usd_bn,
    ROUND(ms.value / 1e9, 1)                    AS ms_revenues_usd_bn,
    CASE WHEN gs.value > ms.value THEN 'GS' ELSE 'MS' END AS higher,
    ROUND(ABS(gs.value - ms.value) / 1e9, 1)    AS difference_usd_bn
FROM financials AS gs
JOIN financials AS ms
  ON ms.metric = gs.metric AND ms.fiscal_year = gs.fiscal_year
WHERE gs.company = 'GS' AND ms.company = 'MS'
  AND gs.metric = 'Revenues' AND gs.fiscal_year = 2024;

-- id: 32
-- Which had higher net income in fiscal year 2023, Citigroup or Bank of America, and by how much?
SELECT
    ROUND(c.value / 1e9, 1)                    AS c_net_income_usd_bn,
    ROUND(bac.value / 1e9, 1)                  AS bac_net_income_usd_bn,
    CASE WHEN c.value > bac.value THEN 'C' ELSE 'BAC' END AS higher,
    ROUND(ABS(c.value - bac.value) / 1e9, 1)   AS difference_usd_bn
FROM financials AS c
JOIN financials AS bac
  ON bac.metric = c.metric AND bac.fiscal_year = c.fiscal_year
WHERE c.company = 'C' AND bac.company = 'BAC'
  AND c.metric = 'NetIncomeLoss' AND c.fiscal_year = 2023;

-- id: 33
-- What percentage of Bank of America's total assets were funded by deposits in fiscal year 2025?
SELECT
    ROUND(d.value / 1e9, 1)           AS deposits_usd_bn,
    ROUND(a.value / 1e9, 1)           AS assets_usd_bn,
    ROUND(100 * d.value / a.value, 1) AS deposits_to_assets_pct
FROM financials AS d
JOIN financials AS a
  ON a.company = d.company AND a.fiscal_year = d.fiscal_year
WHERE d.company = 'BAC' AND d.fiscal_year = 2025
  AND d.metric = 'Deposits' AND a.metric = 'Assets';

-- id: 34
-- How much did Goldman Sachs' revenues grow from fiscal year 2022 to 2025, in percent?
SELECT
    ROUND(y0.value / 1e9, 1)                          AS revenues_2022_usd_bn,
    ROUND(y1.value / 1e9, 1)                          AS revenues_2025_usd_bn,
    ROUND(100 * (y1.value - y0.value) / y0.value, 1)  AS growth_pct
FROM financials AS y0
JOIN financials AS y1
  ON y1.company = y0.company AND y1.metric = y0.metric
WHERE y0.company = 'GS' AND y0.metric = 'Revenues'
  AND y0.fiscal_year = 2022 AND y1.fiscal_year = 2025;

-- id: 35
-- By how many US dollars did JPMorgan Chase's total deposits change from fiscal year 2021 to 2025?
SELECT
    ROUND(y0.value / 1e9, 1)              AS deposits_2021_usd_bn,
    ROUND(y1.value / 1e9, 1)              AS deposits_2025_usd_bn,
    ROUND((y1.value - y0.value) / 1e9, 1) AS change_usd_bn
FROM financials AS y0
JOIN financials AS y1
  ON y1.company = y0.company AND y1.metric = y0.metric
WHERE y0.company = 'JPM' AND y0.metric = 'Deposits'
  AND y0.fiscal_year = 2021 AND y1.fiscal_year = 2025;

-- id: 36
-- What was the percentage change in Morgan Stanley's net income from fiscal year 2024 to 2025?
SELECT
    ROUND(y0.value / 1e9, 1)                          AS net_income_2024_usd_bn,
    ROUND(y1.value / 1e9, 1)                          AS net_income_2025_usd_bn,
    ROUND(100 * (y1.value - y0.value) / y0.value, 1)  AS change_pct
FROM financials AS y0
JOIN financials AS y1
  ON y1.company = y0.company AND y1.metric = y0.metric
WHERE y0.company = 'MS' AND y0.metric = 'NetIncomeLoss'
  AND y0.fiscal_year = 2024 AND y1.fiscal_year = 2025;

-- id: 37
-- Which bank had the largest percentage growth in total assets between fiscal years 2015 and 2025?
SELECT
    y0.company,
    ROUND(y0.value / 1e9, 1)                          AS assets_2015_usd_bn,
    ROUND(y1.value / 1e9, 1)                          AS assets_2025_usd_bn,
    ROUND(100 * (y1.value - y0.value) / y0.value, 1)  AS growth_pct
FROM financials AS y0
JOIN financials AS y1
  ON y1.company = y0.company AND y1.metric = y0.metric
WHERE y0.metric = 'Assets' AND y0.fiscal_year = 2015 AND y1.fiscal_year = 2025
ORDER BY (y1.value - y0.value) / y0.value DESC
LIMIT 1;

-- id: 38
-- What were Truist Financial's total deposits at the end of fiscal year 2025?
SELECT ROUND(value / 1e9, 1) AS deposits_usd_bn
FROM financials
WHERE company = 'TFC' AND metric = 'Deposits' AND fiscal_year = 2025;

-- id: 39
-- What was Citigroup's total stockholders' equity at the end of fiscal year 2023?
SELECT ROUND(value / 1e9, 1) AS equity_usd_bn
FROM financials
WHERE company = 'C' AND metric = 'StockholdersEquity' AND fiscal_year = 2023;

-- id: 40
-- What was Bank of America's net income in fiscal year 2019?
SELECT ROUND(value / 1e9, 1) AS net_income_usd_bn
FROM financials
WHERE company = 'BAC' AND metric = 'NetIncomeLoss' AND fiscal_year = 2019;
