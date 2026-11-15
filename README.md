# Free Model Aggregator

One server that serves the OpenAI Chat Completions API and the Anthropic Messages API over ten
simulated free-tier model providers, in JSON and in server-sent events, for the make.fun bounty
**free-model-aggregator-01** ("The best free AI model aggregator").

## Run

```sh
# In a trial (the env sets these):
MAKEFUN_ENV_URL=http://127.0.0.1:8080 MAKEFUN_PROMPT_FILE=/path/to/prompt.txt node main.ts

# Print the catalog the entry would register, and exit 0:
node main.ts --list

# Outside a trial (no MAKEFUN_PROMPT_FILE / MAKEFUN_ENV_URL): prints {"providers":[]} and exits 0.
```

Node 22+ standard library only; no dependencies, no build step. The entry listens on `0.0.0.0` at
an OS-chosen port, registers it with the env, serves the client's 20 requests, polls `GET /work`,
and exits 0 when the workload is done.

## Layout

- `main.ts` — the server: brief + catalog probe, rating-ranked routing with budget spreading,
  failover, the two client wires, register/poll/exit.
- `lib/providers.ts` — the brief's three documents (provider directory, quality table, workload
  manifest), the catalog probe, the catalog builder, and the planned per-provider call capacities.
- `lib/wire.ts` — all six client×upstream translations (OpenAI/Anthropic client × OpenAI/Anthropic/
  Gemini provider), JSON and SSE, with stop-reason mapping and exact usage passthrough.
- `LICENSE` — MIT (`SPDX-License-Identifier: MIT`).

## How it routes

Every candidate is a (provider, model) pair that is free and meets the request's requirements. A
JSON-mode request on the OpenAI wire never routes to an anthropic-wire provider, whose wire cannot
carry JSON mode. Candidates rank by the quality table's `rating_milli`, then by how clean the
provider still is (fewer refusals first), then by planned capacity left, then by provider/model id —
a pure function of the brief and the answers seen so far, so a replay sees the same choices.

The plan spreads the workload instead of burning one provider dry: the top model (OpenRouter's
`z-ai/glm-5.3-flash:free`, 1475) cannot serve all 17 requests that want it on a ~10-call budget, so
the router leans on the next rungs (1470 on Z.ai and Cloudflare, 1465 on SambaNova, 1462 on
OpenRouter's DeepSeek, 1458 on Gemini). Gemini-3.8-flash is the only tools-plus-vision model, so the
tools-plus-vision and vision requests take it first. Planned capacities are estimates from the kit's
generator (`program/generator/draw.mjs`, each provider's `calls_min..calls_max`, midpoint less one
call for the provider's failure schedule); the router adapts at runtime from every refusal it sees.

## How it fails over

A provider's refusal (402 when its free allowance is gone, 429 everywhere else, or a 5xx) is
per-provider, not per-model — but only one call number is poisoned by a scheduled fail, so the
router gives a refused provider one more try, on its next-best model, before moving down the
ladder; that second call usually answers, at a rating barely below the first pick. A refusal counts
against the provider's budget, so the router counts it too. Three refusals in a row mean the budget
is spent or the outage has begun, and the provider is dead; two in a row kill it only once it has
made enough calls to plausibly be spent (a lone pair of adjacent scheduled fails leaves it alive).
A call that never reaches the provider spends no budget and is not counted. The router never waits
on a retry-after — it fails over at once to another model, and only reconsiders the refused
provider for a later request.

## How it translates

The translation is lossless by construction: the provider echoes what it parsed (system prompt,
user text with images, tool names, stop sequences, `max_tokens`, JSON mode) and the client compares
that echo with what it sent, so every field is carried across in order. The provider's answer text
returns whole, its usage counts exactly as its wire reports them, and its stop reason is mapped
across the wires (`end_turn` to `stop`, `max_tokens` to `length`, `tool_use` to `tool_calls`,
`refusal` to `content_filter`, and the inverses). An OpenAI stream asks the provider for its usage
chunk (`stream_options: {include_usage: true}`); without it the provider's stream carries no usage
and the client's usage check has nothing to compare.

## Against the real services

Provider config (keys, base URLs) lives in the brief's directory, never in this code. A key the
brief leaves empty is read from the environment variable the directory names (`key_env`), and a
directory entry may give an absolute `base_url` instead of a path — so the same code runs against
the real services with real keys and base URLs.

## Grade locally

```sh
KIT=~/workspace/makefun/free-model-aggregator-01-kit
BOT=~/workspace/makefun/free-model-aggregator-01
# Full kit gate (process driver; no Docker needed):
node $KIT/cli/makefun.mjs grade-local --kit $KIT --bot $BOT --driver process --instances 3
```

Or by hand, one practice trial (env port 18120, grade dir of your choice):

```sh
G=~/workspace/makefun/grade-agg-trial0
mkdir -p $G
MAKEFUN_ENV_DATA_DIR=$KIT/practice/trial-0/input/env MAKEFUN_STATE_DIR=$G \
  MAKEFUN_ENV_PORT=18120 node $KIT/program/env/server.mjs > $G/env.log 2>&1 &
# wait for $G/ready, then:
MAKEFUN_ENV_URL=http://127.0.0.1:18120 \
  MAKEFUN_PROMPT_FILE=$KIT/practice/trial-0/input/prompt.txt \
  timeout 120 node $BOT/main.ts > $G/entry.log 2>&1
# SIGTERM only the env's node (MainThread) PID, then score:
MAKEFUN_OUTPUT_DIR=$G/export MAKEFUN_REFERENCE_DIR=$KIT/practice/trial-0/reference \
  MAKEFUN_SCORES_DIR=$G node $KIT/program/scorer/score.mjs
cat $G/scores.json
```

## Local grades

| Trial | Score | Quality | Served | Wire | Failover | Catalog | Upstream calls | Notes |
|-------|-------|---------|--------|------|----------|---------|----------------|-------|
| trial-0 | 99 | 99 | 100 | 100 | 100 | 100 | 28 | 20/20 valid, 233/233 checks; failover 6/6 |
| trial-1 | 99 | 99 | 100 | 100 | 100 | 100 | 27 | 20/20 valid, 233/233 checks; failover 6/6 |
| trial-2 | 99 | 99 | 100 | 100 | 100 | 100 | 30 | 20/20 valid, 233/233 checks; failover 8/8 |
| mean | 99 | 99 | 100 | 100 | 100 | 100 | 28.3 | kit `grade-local --driver process --instances 3`: 3/3 finished, 0 timeout, 0 nondeterministic; replay matched byte-for-byte on all three |
