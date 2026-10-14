-- Embeddings are content-derived (inversion can reconstruct approximate
-- text from vectors), so they join every other content column inside the
-- ciphertext boundary. Legacy plaintext vectors are encrypted in place by
-- the app's background pass, then nulled.

ALTER TABLE memory_chunks ADD COLUMN IF NOT EXISTS embedding_enc BYTEA;

ALTER TABLE memory_chunks ALTER COLUMN embedding DROP NOT NULL;
ALTER TABLE memory_chunks ALTER COLUMN norm DROP NOT NULL;

CREATE INDEX IF NOT EXISTS memory_chunks_legacy_idx
    ON memory_chunks (id)
    WHERE embedding IS NOT NULL AND embedding_enc IS NULL;
