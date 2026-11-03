# Instrumented Retrieval Audit of 37 Stage 1 Candidate-Pool Failures

**Generated:** 2026-11-03 08:14:22 UTC
**Audited Questions:** 37 Stage 1 candidate-pool failures.

## 1. Aggregate Path Recall & Fusion Summary

| Retrieval Path | Questions with Gold Hit | Hit Rate (%) | Mean Hit Rank | Median Hit Rank |
| :--- | :---: | :---: | :---: | :---: |
| `path1_bm25` | 0/37 | 0.0% | N/A | N/A |
| `path2_dense` | 4/37 | 10.81% | 19.5 | 20 |
| `path3_metadata` | 1/37 | 2.7% | 14.0 | 14 |
| `path4_hyde` | 7/37 | 18.92% | 21.29 | 23 |
| **Union of All 4 Paths** | **12/37** | **32.43%** | - | - |
| **Fused RRF Top-40** | **0/37** | **0.0%** | - | - |

## 2. Key Forensic Insights

- **Zero-Path Retrieval Failures:** In 25 of 37 cases, none of the 4 paths retrieved gold evidence within their configured retrieval depth.
- **Fusion Displacement:** In 12 of 37 cases, gold evidence was retrieved by at least one individual path but was displaced outside the top-40 by RRF.

## 3. Question-Level Retrieval Matrix

| Question ID | Target Doc | Pages | BM25 Hit | Dense Hit | Meta Hit | HyDE Hit | Union Hit | RRF Hit |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `financebench_id_01226` | `3M_2022_10K` | 27 | None | None | None | None | False | False |
| `financebench_id_01865` | `3M_2022_10K` | 25 | None | None | None | None | False | False |
| `financebench_id_00438` | `ADOBE_2022_10K` | 54 | None | None | None | None | False | False |
| `financebench_id_01319` | `AES_2022_10K` | 132 | None | None | None | None | False | False |
| `financebench_id_10420` | `AES_2022_10K` | 130;132 | None | None | None | 23 | True | False |
| `financebench_id_00799` | `AMCOR_2023_10K` | 52 | None | None | None | None | False | False |
| `financebench_id_01079` | `AMCOR_2023_10K` | 64 | None | None | None | 31 | True | False |
| `financebench_id_00684` | `AMCOR_2023_10K` | 50 | None | None | None | None | False | False |
| `financebench_id_01928` | `AMCOR_2023Q4_EARNINGS` | 12 | None | 15 | None | None | True | False |
| `financebench_id_00563` | `AMD_2022_10K` | 48 | None | None | None | 9 | True | False |
| `financebench_id_01028` | `AMERICANEXPRESS_2022_10K` | 155 | None | None | None | None | False | False |
| `financebench_id_00723` | `AMERICANEXPRESS_2022_10K` | 96 | None | None | None | None | False | False |
| `financebench_id_01351` | `AMERICANEXPRESS_2022_10K` | 44 | None | None | None | None | False | False |
| `financebench_id_00685` | `BESTBUY_2023_10K` | 40 | None | None | None | None | False | False |
| `financebench_id_01077` | `BESTBUY_2023_10K` | 51 | None | None | None | None | False | False |
| `financebench_id_04660` | `BLOCK_2016_10K` | 68 | None | None | None | None | False | False |
| `financebench_id_01290` | `BOEING_2022_10K` | 8;10;14 | None | None | None | None | False | False |
| `financebench_id_10130` | `CORNING_2020_10K` | 70;72 | None | None | None | 11 | True | False |
| `financebench_id_05915` | `CVSHEALTH_2018_10K` | 302;304 | None | None | None | 12 | True | False |
| `financebench_id_04103` | `GENERALMILLS_2019_10K` | 53;55 | None | None | None | None | False | False |
| `financebench_id_03471` | `GENERALMILLS_2020_10K` | 50 | None | None | None | None | False | False |
| `financebench_id_00956` | `JOHNSON_JOHNSON_2022_10K` | 28 | None | None | 14 | None | True | False |
| `financebench_id_00669` | `JOHNSON_JOHNSON_2022_10K` | 34 | None | None | None | None | False | False |
| `financebench_id_00711` | `JOHNSON_JOHNSON_2022_10K` | 46;47 | None | None | None | None | False | False |
| `financebench_id_01484` | `JOHNSON_JOHNSON_2022Q4_EARNINGS` | 2 | None | 29 | None | None | True | False |
| `financebench_id_00299` | `JPMORGAN_2021Q1_10Q` | 19 | None | None | None | 28 | True | False |
| `financebench_id_02119` | `JPMORGAN_2021Q1_10Q` | 6 | None | None | None | None | False | False |
| `financebench_id_00206` | `JPMORGAN_2022_10K` | 3 | None | None | None | 35 | True | False |
| `financebench_id_00394` | `JPMORGAN_2022Q2_10Q` | 21 | None | None | None | None | False | False |
| `financebench_id_04700` | `MICROSOFT_2016_10K` | 52 | None | None | None | None | False | False |
| `financebench_id_00552` | `MICROSOFT_2023_10K` | 60 | None | None | None | None | False | False |
| `financebench_id_00735` | `PEPSICO_2022_10K` | 26 | None | None | None | None | False | False |
| `financebench_id_01328` | `PEPSICO_2022_10K` | 78 | None | 14 | None | None | True | False |
| `financebench_id_00603` | `ULTABEAUTY_2023Q4_EARNINGS` | 3 | None | None | None | None | False | False |
| `financebench_id_00605` | `ULTABEAUTY_2023Q4_EARNINGS` | 3 | None | 20 | None | None | True | False |
| `financebench_id_00216` | `VERIZON_2022_10K` | 56 | None | None | None | None | False | False |
| `financebench_id_00566` | `VERIZON_2022_10K` | 77 | None | None | None | None | False | False |
