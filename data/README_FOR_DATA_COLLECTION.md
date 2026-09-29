# Medical Drug Dataset

## Overview

This is the final verified medical drug dataset built from:

- RxNorm
- DailyMed
- FDA NDC data

MedlinePlus was intentionally skipped from the final release.

The dataset is designed to be compact, relational, searchable,
machine-learning friendly, and suitable for downstream medical AI
applications.


## Final Files

| File | Rows | Approx. Size | Purpose |
|---|---:|---:|---|
| medical_drug_master.csv | 104,450 | 44.03 MB | Main drug/label dataset |
| medical_drug_ingredients.csv | 184,343 | 7.43 MB | RxNorm ingredient relationships |
| medical_ndc_reference.csv | 234,750 | 40.25 MB | FDA/NDC reference |
| final_qc_report.txt | - | - | Final QC report |


## Total Dataset Size

Approximately 91.71 MB.


## Architecture

The dataset consists of three related tables:

    medical_drug_master
            |
            +---- primary_rxcui ----> medical_drug_ingredients
            |
            +---- SETID -----------> medical_ndc_reference


## Main Identifiers

Master:

    SETID

RxNorm:

    primary_rxcui
    RXCUI

FDA/NDC:

    NDC


## Data Sources

### RxNorm

Used for:

- RxCUI
- drug concepts
- ingredients
- strengths
- identifiers
- terminology relationships

### DailyMed

Used as the main drug/label source.

Provides:

- SETID
- SPL version
- drug name
- RxNorm relationships
- ingredient information
- strength information
- NDC counts
- label metadata

### FDA

Used for NDC-level product verification and metadata.

Provides:

- NDC
- proprietary name
- dosage form
- marketing category
- application number
- product type
- marketing dates
- billing unit
- inactivation/reactivation information
- FDA matching status


## RxNorm Match Status

The final master contains:

MATCHED

The DailyMed record was successfully resolved against the
available RxNorm release.

PARTIALLY_MATCHED

Some RxNorm information was resolved but the complete desired
relationship was not established.

UNRESOLVED_IN_CURRENT_RXNORM_RELEASE

The DailyMed record could not be completely resolved using the
RxNorm release available during processing.

These records are intentionally retained.


## FDA Match Status

Observed values include:

- FDA_MATCHED
- FDA_NOT_MATCHED
- FDA_MATCHED|FDA_NOT_MATCHED
- NULL

FDA_MATCHED|FDA_NOT_MATCHED represents a record containing both
matching and non-matching NDC-level outcomes.

Do not convert this field into a simple boolean without inspecting
the underlying NDC information.


## Final QC

The final cross-file QC passed.

Important results:

- Master duplicate SETIDs: 0
- Missing master SETIDs: 0
- Master RxCUI coverage: 100%
- Ingredient duplicate rows: 0
- Orphan ingredient RxCUIs: 0
- Duplicate NDCs: 0
- Missing NDCs: 0
- Orphan NDC SETIDs: 0
- Orphan NDC RxCUIs: 0


## Recommended Usage

For most applications:

1. Start with medical_drug_master.csv.
2. Join ingredient information using RxCUI.
3. Use medical_ndc_reference.csv for NDC/FDA lookup.
4. Keep the three-table structure.
5. Do not flatten all data into one giant CSV.


## Intended Applications

Possible applications include:

- medicine search
- medicine identification
- RxCUI lookup
- NDC verification
- ingredient lookup
- drug normalization
- prescription processing
- medical NLP
- clinical information retrieval
- medical AI pipelines


## Important Limitation

This dataset is a structured medical information resource.

It must not be treated as a standalone clinical decision-making
system. Clinical applications require appropriate validation,
regulatory compliance, and human oversight.


## Release Status

FINAL DATASET - QC PASSED

The three CSV files should be treated as the frozen dataset release.
