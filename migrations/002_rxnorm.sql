-- RxNorm concept strings used to resolve OCR-extracted drug text to an RxCUI.
--
-- `source` records provenance of every row so an auditor can tell UMLS
-- RXNCONSO.RRF rows apart from rows seeded through the public RxNav API.
CREATE TABLE IF NOT EXISTS rxnorm_concept (
    rxcui      text        NOT NULL,
    tty        text        NOT NULL,
    str        text        NOT NULL,
    normalized text        NOT NULL,
    source     text        NOT NULL CHECK (source IN ('RXNCONSO', 'RXNAV_API')),
    loaded_at  timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (rxcui, tty, str)
);

-- Tier 1 of the resolution ladder: exact, case-insensitive.
CREATE INDEX IF NOT EXISTS rxnorm_concept_lower_str_idx ON rxnorm_concept (lower(str));

-- Tier 2: normalized equality.
CREATE INDEX IF NOT EXISTS rxnorm_concept_normalized_idx ON rxnorm_concept (normalized);

-- Tier 3: trigram similarity.
CREATE INDEX IF NOT EXISTS rxnorm_concept_trgm_idx
    ON rxnorm_concept USING gin (normalized gin_trgm_ops);

-- Ingredient-to-product RxCUI relations, used to verify that an openFDA label
-- really describes the resolved ingredient. openFDA records carry
-- product-level RxCUIs (SCD/SBD), never the ingredient RxCUI, so a name match
-- alone would not prove the label belongs to this drug.
CREATE TABLE IF NOT EXISTS rxnorm_ingredient_product (
    ingredient_rxcui text        NOT NULL,
    product_rxcui    text        NOT NULL,
    source           text        NOT NULL CHECK (source IN ('RXNCONSO', 'RXNAV_API')),
    loaded_at        timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (ingredient_rxcui, product_rxcui)
);

CREATE INDEX IF NOT EXISTS rxnorm_ingredient_product_ingredient_idx
    ON rxnorm_ingredient_product (ingredient_rxcui);
