## 16. Milestone 10 — citation graph, packaging, and deployment

1. Add citation relationships using OpenAlex referenced-work IDs.

2. Use NetworkX before considering a graph database.

3. Containerize only after local installation is stable.

4. Add secrets management, logging, monitoring, rate limiting, retries, and cost budgets.

5. Deploy an API and UI separately only if operational complexity is justified.

6. Citation edges are `citing -> cited` and are induced only over the bounded canonical retrieved pool. Count external references without recursively retrieving them. The graph is descriptive and must not change relevance, evidence, gap generation, confidence, or verification.

7. Quick Search does not construct a graph. Historical results without graph data load as unavailable citation context.

8. Provider budgets are disabled by default, persistent and concurrency-safe when enabled, and apply to administrator analyses independently from application credits. Missing provider token metadata is unavailable, never zero.
