-- Control questions (not in eval/questions.csv) to check whether the sql_run2
-- prompt change generalizes beyond the 15 figure questions used to diagnose it.
-- Proposed 2026-11-24; NOT added to eval/questions.csv until approved.
-- Same conventions as eval/figure_questions.sql: amounts in USD billions and
-- percents rounded to 1 decimal, rounding only in the final SELECT.
-- Run read-only on data/processed/finance.duckdb.

-- id: c1
-- What were Goldman Sachs' total liabilities at the end of fiscal year 2022?
SELECT ROUND(value / 1e9, 1) AS liabilities_usd_bn
FROM financials
WHERE company = 'GS' AND metric = 'Liabilities' AND fiscal_year = 2022;

-- id: c2
-- Which had higher stockholders' equity in fiscal year 2021, Bank of America or Citigroup, and by how much?
SELECT
    ROUND(bac.value / 1e9, 1)                   AS bac_equity_usd_bn,
    ROUND(c.value / 1e9, 1)                     AS c_equity_usd_bn,
    CASE WHEN bac.value > c.value THEN 'BAC' ELSE 'C' END AS higher,
    ROUND(ABS(bac.value - c.value) / 1e9, 1)    AS difference_usd_bn
FROM financials AS bac
JOIN financials AS c
  ON c.metric = bac.metric AND c.fiscal_year = bac.fiscal_year
WHERE bac.company = 'BAC' AND c.company = 'C'
  AND bac.metric = 'StockholdersEquity' AND bac.fiscal_year = 2021;

-- id: c3
-- What percentage of JPMorgan Chase's total assets were liabilities in fiscal year 2020?
SELECT
    ROUND(l.value / 1e9, 1)           AS liabilities_usd_bn,
    ROUND(a.value / 1e9, 1)           AS assets_usd_bn,
    ROUND(100 * l.value / a.value, 1) AS liabilities_to_assets_pct
FROM financials AS l
JOIN financials AS a
  ON a.company = l.company AND a.fiscal_year = l.fiscal_year
WHERE l.company = 'JPM' AND l.fiscal_year = 2020
  AND l.metric = 'Liabilities' AND a.metric = 'Assets';

-- id: c4
-- By how many US dollars did Morgan Stanley's total assets change from fiscal year 2018 to 2022?
SELECT
    ROUND(y0.value / 1e9, 1)              AS assets_2018_usd_bn,
    ROUND(y1.value / 1e9, 1)              AS assets_2022_usd_bn,
    ROUND((y1.value - y0.value) / 1e9, 1) AS change_usd_bn
FROM financials AS y0
JOIN financials AS y1
  ON y1.company = y0.company AND y1.metric = y0.metric
WHERE y0.company = 'MS' AND y0.metric = 'Assets'
  AND y0.fiscal_year = 2018 AND y1.fiscal_year = 2022;

-- id: c5
-- Which bank had the smallest percentage growth in total deposits between fiscal years 2016 and 2024?
SELECT
    y0.company,
    ROUND(y0.value / 1e9, 1)                          AS deposits_2016_usd_bn,
    ROUND(y1.value / 1e9, 1)                          AS deposits_2024_usd_bn,
    ROUND(100 * (y1.value - y0.value) / y0.value, 1)  AS growth_pct
FROM financials AS y0
JOIN financials AS y1
  ON y1.company = y0.company AND y1.metric = y0.metric
WHERE y0.metric = 'Deposits' AND y0.fiscal_year = 2016 AND y1.fiscal_year = 2024
ORDER BY (y1.value - y0.value) / y0.value ASC
LIMIT 1;
