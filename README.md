# hermes-agentmemory

A drop-in [Hermes Agent](https://github.com/NousResearch/hermes-agent) memory plugin built on [agentmemory](https://github.com/MukundaKatta/agentmemory).

Pull-model episodic memory with real deletes and an audit trace. The point: Hermes Agent is good at remembering. This plugin gives it a memory layer that takes deletion seriously.

## Why another memory plugin?

Hermes ships with several first-class memory backends (Mem0, Honcho, Hindsight, etc.). They consolidate in the background, which is the dominant pattern in agentic memory right now. That makes recall cheap and fast at the cost of two things:

1. **Deletes are not always real.** Once an episode is baked into a derived summary, removing the original event leaves the summary intact.
2. **Memory injection is opaque.** Background prefetch happens off the hot path; the user does not see exactly which past events were used until something goes wrong.

`agentmemory` flips both. It does no background work, every write is synchronous, deletes are immediate and complete, and every prefetch writes a trace record (`event_ids` + `summary` + `prompt`) to `$HERMES_HOME/agentmemory/trace.jsonl` so the user can audit what entered the prompt.

## Install

```bash
# from the Hermes repo root
mkdir -p plugins/memory/agentmemory
cp -r path/to/hermes-agentmemory/* plugins/memory/agentmemory/

# activate
hermes config set memory.provider agentmemory

# set Anthropic key for the summarizer
export ANTHROPIC_API_KEY=...
```

## Configuration

Environment variables:

| Var | Default | Purpose |
| --- | --- | --- |
| `ANTHROPIC_API_KEY` | (required) | key for the on-demand summarizer |
| `AGENTMEMORY_MODEL` | `claude-sonnet-4-5-20251022` | Claude model id |
| `AGENTMEMORY_TOP_K` | `5` | events to retrieve per prefetch |
| `AGENTMEMORY_MAX_TOKENS` | `300` | summary token budget |
| `AGENTMEMORY_TRACE_LOG` | `$HERMES_HOME/agentmemory/trace.jsonl` | where to append audit records |

## Tools the agent can call

- `agentmemory_recall(query, top_k?)` — surface the top matching past events plus the event ids used.
- `agentmemory_forget(session_id?, event_id?)` — real delete. No tombstone, no derived artifact left behind.
- `agentmemory_drift()` — rolling-window retrieval-quality state, useful when recall starts feeling stale.

## Auditing what the model saw

```bash
tail -f ~/.hermes/agentmemory/trace.jsonl
```

Every prefetch produces one JSON line: `intent`, `event_ids`, `summary`, and the live `drift` snapshot.

## Trade-off, honestly

The first turn of every new session pays a 200ms-2s tax for the on-demand summary because there is no background pre-warming. In exchange you get:

- deletes that are real and immediate
- no quality decay from a smaller summarizer model (the summarizer is the same Claude family the agent uses)
- a trace file the user can audit without touching the agent
- < 600 lines of Python you can read end-to-end

For a self-hosted agent that markets itself as "the agent that grows with you", auditable memory is the part that lets growth stay reversible.

## License

MIT.

## See also

- The library this wraps: [github.com/MukundaKatta/agentmemory](https://github.com/MukundaKatta/agentmemory) (npm: `@mukundakatta/agentmemory`)
- The design rationale: [Self-improving agents need to forget too](https://dev.to/mukundakatta/self-improving-agents-need-to-forget-too-a-memory-primitive-for-hermes-agent-kbd) (dev.to)
- The "why I refused to clone Dreaming" companion: [dev.to/mukundakatta](https://dev.to/mukundakatta/why-i-refused-to-build-a-dreaming-clone-for-oss-claude-2631)
