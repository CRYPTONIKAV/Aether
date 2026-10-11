-- Vector memory: chunked, sourced, embedded pieces of every event, so
-- retrieval is semantic rather than a recency window. One event -> many
-- chunks; each chunk keeps the event's identity (event_id, source, kind)
-- plus where it sat inside its document (filename, location), so results
-- from different documents never merge.
--
-- Chunk text is encrypted like event payloads (text_enc, AAD per row).
-- The embedding itself is derived data, stored plain: it cannot be read
-- back as text, and cosine scoring has to load it anyway.

ALTER TABLE events ADD COLUMN IF NOT EXISTS indexed_at TIMESTAMPTZ;

CREATE TABLE IF NOT EXISTS memory_chunks (
    id          BIGSERIAL PRIMARY KEY,
    event_id    BIGINT      NOT NULL REFERENCES events(id) ON DELETE CASCADE,
    source      TEXT        NOT NULL,
    kind        TEXT        NOT NULL,
    filename    TEXT        NOT NULL DEFAULT '',
    location    TEXT        NOT NULL DEFAULT '',
    text_enc    BYTEA       NOT NULL,
    embedding   REAL[]      NOT NULL,
    norm        REAL        NOT NULL,
    embed_model TEXT        NOT NULL,
    occurred_at TIMESTAMPTZ NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS memory_chunks_event_idx ON memory_chunks (event_id);
CREATE INDEX IF NOT EXISTS memory_chunks_model_idx ON memory_chunks (embed_model);
