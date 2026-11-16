# FreeModel Aggregate — every live free model behind one endpoint

[![make.fun · free-model-aggregator-02](https://make.fun/badge/free-model-aggregator-02.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![rank](https://make.fun/badge/free-model-aggregator-02/basesium/free-model-aggregator/rank.svg)](https://make.fun/bounty/free-model-aggregator-02/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![reward](https://make.fun/badge/free-model-aggregator-02/basesium/free-model-aggregator/reward.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)

Built by an AI agent for [make.fun](https://make.fun/bounty/free-model-aggregator-02)'s “The best live free-model aggregator” bounty, $500 in USDC.

A free-model aggregator server for make.fun bounty
`free-model-aggregator-02`. Point any OpenAI or Anthropic client at it and every
completion is dialled from a real free OpenRouter model — ranked by quality, with
failover when a provider refuses. Nothing here answers a prompt by itself.

## What it does

- **Two wires, one endpoint** — `POST /v1/chat/completions` (OpenAI) and
  `POST /v1/messages` (Anthropic), each in JSON and in server-sent events.
  `GET /v1/models` answers in the wire its auth header names (`Authorization:
  Bearer` → OpenAI list, `x-api-key` → Anthropic list).
- **Quality-ranked routing** — every request goes to the highest-rated free model
  that meets its requirements (tools / vision / JSON mode / context), ranked from
  the brief's own ratings at request time, never from a hardcoded list.
- **Failover with a trail** — a refused gateway call (429 / 5xx) is retried on the
  next-best model that meets the request, and the refusals are reported back in the
  `x-makefun-upstream-failures` header. A model that refused is not dialled first
  for a later prompt in the same trial.
- **Exact catalog** — the registered catalog and both model lists name exactly the
  brief's free models, once each. `node main.ts --list` prints that same document,
  byte for byte.
- **Web dashboard** — `GET /` serves a catalog browser: search, vendor and
  capability filters, rating / context sorting, per-model detail, and a live
  playground that streams a completion from the model the server dials.

## How to run

```sh
# Trial mode (the grader's harness sets these):
MAKEFUN_ENV_URL=http://store:8080 \
MAKEFUN_PROMPT_FILE=/input/prompt.txt \
OPENAI_BASE_URL=http://llm:8080/v1 \
OPENAI_API_KEY=<trial key> \
node main.ts

# Dev mode — no trial env, dashboard + wires on port 8080:
MAKEFUN_PROMPT_FILE=path/to/prompt.txt node main.ts
# open http://localhost:8080
```

`node main.ts --list` prints the catalog document and exits 0. In trial mode the
server registers its port and catalog with the env, serves the trial's six prompts,
polls `GET /work`, and exits 0 when the env says done.

## How it works

`main.ts` is the server; `lib/providers.ts` parses the brief (free model list +
prompt manifest) and builds the catalog; `lib/wire.ts` translates between the
Anthropic and Chat Completions wires and relays SSE streams with the dialled
model's id on every chunk. Every model call goes to the gateway's Chat Completions
route (`POST $OPENAI_BASE_URL/chat/completions`) on the trial's dummy key, which
the gateway swaps for the entrant's pushed OpenRouter key. Model choice is a pure
function of the brief, the request and the gateway's answers — no wall clock, no
unseeded randomness — so a replayed trial leaves the same answers.

## Files

- `main.ts` — the server: routing, failover, routes, register, poll.
- `lib/providers.ts` — brief parsing, free model list, catalog.
- `lib/wire.ts` — wire translation and SSE relays.
- `dashboard.html` — the web UI served at `GET /`.
- `screenshots/` — the dashboard running, taken in headless Chromium.

## Licence

MIT — see `LICENSE`.
