-- Cache of openFDA drug-label responses, keyed by the ingredient RxCUI the
-- label was accepted for. `payload` holds only the sections RxPlain quotes,
-- plus the SPL identifiers used as provenance refs.
--
-- `accepted_product_rxcuis` records which product RxCUIs matched between the
-- label and rxnorm_ingredient_product, so the join can be re-audited later.
CREATE TABLE IF NOT EXISTS openfda_label_cache (
    ingredient_rxcui        text        PRIMARY KEY,
    spl_id                  text        NOT NULL,
    spl_set_id              text        NOT NULL,
    accepted_product_rxcuis text[]      NOT NULL,
    payload                 jsonb       NOT NULL,
    fetched_at              timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS openfda_label_cache_fetched_at_idx
    ON openfda_label_cache (fetched_at);
