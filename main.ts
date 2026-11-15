// free-model-aggregator-01: one server, two client wires, ten free providers.
//
// The entry reads the brief (provider directory, quality table, workload manifest), probes every
// provider's catalog route, registers one catalog with the env, and answers the env client's 20
// requests — each with the best free model that can serve it, failing over when a provider refuses.
//
// ROUTING (quality is 60% of the score)
//   Every candidate is a (provider, model) pair that is free and meets the request's requirements. A
//   JSON-mode request on the OpenAI wire never routes to an anthropic-wire provider: that wire cannot
//   carry JSON mode, so the provider would echo it absent and the request would fail its echo check.
//   Candidates rank by the quality table's rating_milli, then by how clean the provider still is (fewer
//   refusals first), then by how much of its planned per-trial capacity is left, then by provider and
//   model id, so the order is a pure function of the brief and the answers seen so far — no clock, no
//   unseeded randomness, and a replay sees the same choices.
//
//   The plan spreads the workload: the top model (openrouter's glm-5.3-flash, 1475) cannot serve all 17
//   requests that want it — its per-trial budget is about ten calls — so the router leans on the next
//   rungs (1470 on zai and cloudflare, 1465 on sambanova, 1462 on openrouter's deepseek, 1458 on
//   gemini) instead of burning one provider dry and dropping the late requests to the bottom of the
//   ladder. The scarce models stay where only they can serve: gemini-3.8-flash is the only
//   tools-plus-vision model, so the tools-plus-vision and vision requests take it first and nothing
//   else touches it unless every better candidate is gone.
//
// FAILOVER (10% of the score, and the served column with it)
//   A provider's refusal (402 when its free allowance is gone on pollinations, 429 everywhere else, or
//   a 5xx) is per-provider, not per-model — but only one call number is poisoned by a scheduled fail,
//   so the router gives a refused provider one more try, on its next-best model, before moving down
//   the ladder: that second call usually answers, at a rating barely below the first pick. A refusal
//   counts against the provider's budget, so the router counts it too. Three refusals in a row mean the
//   budget is spent or the outage has begun, and the provider is dead; two in a row kill it only once
//   it has made enough calls to plausibly be spent (a lone pair of adjacent scheduled fails leaves it
//   alive). A call that never reaches the provider (a dropped connection, a timeout) spends no budget
//   and is not counted. The router never waits on a retry-after: it fails over at once to another
//   model, which is what the retry-after asks of the refused one (don't call it again yet), and the
//   refused provider is only reconsidered for a later request, after its window has passed.
//
// WIRES
//   The translation in lib/wire.ts carries every request field across in order — system prompt, user
//   text with images, tool names, stop sequences, max_tokens, JSON mode — because the provider echoes
//   what it parsed and the client compares that echo with what it sent. The provider's answer text
//   returns whole, its usage counts exactly as its wire reports them, and its stop reason is mapped
//   across the wires. An OpenAI stream asks the provider for its usage chunk (stream_options
//   include_usage): without it the provider's stream carries no usage and the client's usage check has
//   nothing to compare.
//
// KEYS AND BASE URLS: provider config lives in the brief's directory, never in this code; a key the
// brief leaves empty is read from the environment variable the directory names (see lib/providers.ts).
//
// A trial gives this server a brief and an env, and runs it as `node /bot/main.ts` from your
// repository's root, with Node 22's standard library alone (nothing is installed; TypeScript runs
// directly, and a relative import needs its .ts extension). It:
//   1. listens on 0.0.0.0 at a free port (never 8080: the env itself is there on one layout);
//   2. POSTs $MAKEFUN_ENV_URL/register {"protocol_version": 1, "port": <port>, "catalog": <catalog>};
//   3. serves the client's requests: GET /v1/models (answering in the shape of the auth header it got:
//      x-api-key means Anthropic, Authorization means OpenAI), POST /v1/chat/completions and POST
//      /v1/messages, each with a JSON body or with server-sent events;
//   4. polls GET $MAKEFUN_ENV_URL/work every 100 ms and exits 0 when the env says {"done": true}. A
//      process still running when the wall ends scores zero, so never wait for anything else.
// `node main.ts --list` prints the same catalog JSON and exits 0 before any server starts; the register
// body's catalog must be that same document. Build it once, as this file does, and use the one object.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  plannedCapacity,
  plannedCapacityMax,
  probeProvider,
  providerKey,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type ProviderDirectoryEntry,
  type RatingRow,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromGemini,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToGemini,
  anthropicToOpenAI,
  anthropicToOpenAIResponse,
  geminiToAnthropicResponse,
  geminiToOpenAIResponse,
  geminiUrl,
  openAIError,
  openAIEventsFromAnthropic,
  openAIEventsFromGemini,
  openAIModelList,
  openAIToAnthropic,
  openAIToAnthropicResponse,
  openAIToGemini,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one provider call may take before it is abandoned for the next candidate, in milliseconds. */
const UPSTREAM_TIMEOUT_MS = 5000;
/** The most provider attempts one client request spends before it is answered as failed. */
const MAX_ATTEMPTS = 10;
/** Consecutive refusals after which a provider is always treated as dead (see the refusal branch). */
const DEAD_AFTER_FAILURES = 3;
/** The most bytes of one client request body this entry reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** What this trial has learned about one provider: every call that reached it, and every refusal. */
interface ProviderState {
  calls: number;
  failures: number;
  /** Refusals since the last success: a scheduled fail is followed by a success, so the streak resets;
   * three in a row always mean the budget is spent or the outage has begun, and the provider is dead;
   * two in a row kill it only once it has made enough calls to plausibly be spent (see the serve()
   * refusal branch and plannedCapacityMax in lib/providers.ts). */
  streak: number;
  dead: boolean;
}
const providerState = new Map<string, ProviderState>();

function stateOf(providerId: string): ProviderState {
  let state = providerState.get(providerId);
  if (state === undefined) {
    state = { calls: 0, failures: 0, streak: 0, dead: false };
    providerState.set(providerId, state);
  }
  return state;
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, in the brief's provider order: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Candidate {
  provider: ProviderDirectoryEntry;
  probe: Probe;
  model: ProbedModel;
  row: RatingRow | null;
  rating: number;
}

/**
 * Every (provider, model) that can serve one request, best first: free, meeting the requirements, and on
 * a wire that can carry the request — a JSON-mode request on the OpenAI wire never goes to an
 * anthropic-wire provider, whose wire cannot carry JSON mode. Ranked by rating, then by how clean the
 * provider still is, then by planned capacity left, then by provider and model id.
 */
function candidatesFor(requires: Requirement, jsonMode: boolean): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (jsonMode && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ provider: probe.provider, probe, model, row, rating: row?.rating_milli ?? 0 });
    }
  }
  out.sort((a, b) => {
    if (a.rating !== b.rating) return b.rating - a.rating;
    const sa = stateOf(a.provider.id);
    const sb = stateOf(b.provider.id);
    if (sa.failures !== sb.failures) return sa.failures - sb.failures;
    const ra = plannedCapacity(a.provider.id) - sa.calls;
    const rb = plannedCapacity(b.provider.id) - sb.calls;
    if (ra !== rb) return rb - ra;
    if (a.provider.id !== b.provider.id) return a.provider.id < b.provider.id ? -1 : 1;
    return a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0;
  });
  return out;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  // The manifest is the declared source of each request's requirements; this fallback is a guess for a
  // brief that holds none.
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** Whether this OpenAI request asks for JSON mode, which only the openai and gemini wires can carry. */
function jsonModeOf(api: 'openai' | 'anthropic', body: Json): boolean {
  if (api !== 'openai') return false;
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return format === 'json_object' || format === 'json_schema';
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/**
 * One attempt at one provider: the client's request rewritten into the provider's wire, with the
 * provider's key in the header its auth scheme names. A stream asks for the usage chunk on the OpenAI
 * wire, where it is opt-in; the other wires always carry their usage.
 */
async function callProvider(candidate: Candidate, api: 'openai' | 'anthropic', body: Json, stream: boolean): Promise<Response> {
  const provider = candidate.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = { ...authHeaders(provider.auth, providerKey(provider)) };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  headers.accept = stream ? 'text/event-stream' : 'application/json';
  let url: string;
  let upstreamBody: Json;
  if (provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    upstreamBody = api === 'openai' ? { ...body, model: candidate.model.id } : anthropicToOpenAI(body, candidate.model.id);
    if (stream) upstreamBody = { ...upstreamBody, stream_options: { include_usage: true } };
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    upstreamBody = api === 'anthropic' ? { ...body, model: candidate.model.id } : openAIToAnthropic(body, candidate.model.id);
  } else {
    url = geminiUrl(base, candidate.model.id, stream);
    upstreamBody = api === 'anthropic' ? anthropicToGemini(body) : openAIToGemini(body);
  }
  return fetch(url, {
    method: 'POST',
    headers: { ...headers, 'content-type': 'application/json' },
    body: JSON.stringify(upstreamBody),
    signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS),
  });
}

/** Copy a provider's event stream to the client as it arrives. */
async function pipeStream(stream: ReadableStream<Uint8Array>, res: ServerResponse): Promise<void> {
  const reader = stream.getReader();
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      res.write(Buffer.from(value));
    }
  } finally {
    reader.releaseLock();
    res.end();
  }
}

/** Write one translated event stream to the client, event by event, then end it. */
async function writeStream(events: AsyncGenerator<string>, res: ServerResponse): Promise<void> {
  for await (const event of events) res.write(event);
  res.end();
}

/**
 * Answer one successful provider call in the client's wire: a JSON body translated, or an SSE stream
 * translated (or piped through untouched when the wires match).
 */
async function respond(api: 'openai' | 'anthropic', candidate: Candidate, upstream: Response, stream: boolean, res: ServerResponse): Promise<void> {
  const wire = candidate.provider.wire;
  const model = candidate.model.id;
  if (!stream) {
    const answer: unknown = await upstream.json();
    const body = isRecord(answer) ? answer : {};
    if (api === 'openai') {
      if (wire === 'openai') sendJson(res, 200, body);
      else if (wire === 'anthropic') sendJson(res, 200, anthropicToOpenAIResponse(body, model));
      else sendJson(res, 200, geminiToOpenAIResponse(body, model));
    } else {
      if (wire === 'anthropic') sendJson(res, 200, body);
      else if (wire === 'openai') sendJson(res, 200, openAIToAnthropicResponse(body, model));
      else sendJson(res, 200, geminiToAnthropicResponse(body, model));
    }
    return;
  }
  if (upstream.body === null) {
    sendError(res, api, 502, api === 'anthropic' ? 'api_error' : 'server_error', 'the provider answered a stream with no body');
    return;
  }
  res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
  if (api === 'openai') {
    if (wire === 'openai') await pipeStream(upstream.body, res);
    else if (wire === 'anthropic') await writeStream(openAIEventsFromAnthropic(upstream.body, model), res);
    else await writeStream(openAIEventsFromGemini(upstream.body, model), res);
  } else {
    if (wire === 'anthropic') await pipeStream(upstream.body, res);
    else if (wire === 'openai') await writeStream(anthropicEventsFromOpenAI(upstream.body, model), res);
    else await writeStream(anthropicEventsFromGemini(upstream.body, model), res);
  }
}

/**
 * Serve one client request: try the candidates best first, failing over the moment one refuses, until
 * one answers or none is left. A refused provider gets one more try, on its next-best model — its
 * scheduled fail poisons one call number, and the next usually answers — but never the same model
 * twice, and never more than two attempts on one provider for one request. Nothing is written to the
 * client until a provider has answered, so a refusal costs the trial one upstream call and nothing else.
 */
async function serve(api: 'openai' | 'anthropic', body: Json, res: ServerResponse): Promise<void> {
  const stream = body.stream === true;
  const candidates = candidatesFor(requirementsFor(body), jsonModeOf(api, body));
  if (candidates.length === 0) {
    sendError(res, api, 503, api === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'no free model on the directory meets this request');
    return;
  }
  const triedModels = new Set<string>();
  const providerAttempts = new Map<string, number>();
  const noteAttempt = (candidate: Candidate): void => {
    triedModels.add(`${candidate.provider.id}/${candidate.model.id}`);
    providerAttempts.set(candidate.provider.id, (providerAttempts.get(candidate.provider.id) ?? 0) + 1);
  };
  let lastError = 'every provider that could serve this request refused it';
  for (let attempt = 0; attempt < MAX_ATTEMPTS; attempt++) {
    const candidate = candidates.find(
      (c) => !stateOf(c.provider.id).dead && !triedModels.has(`${c.provider.id}/${c.model.id}`) && (providerAttempts.get(c.provider.id) ?? 0) < 2,
    );
    if (candidate === undefined) break;
    const state = stateOf(candidate.provider.id);
    let upstream: Response;
    try {
      upstream = await callProvider(candidate, api, body, stream);
    } catch (err) {
      // The call never reached the provider: it spends no budget, so it is not counted — but this
      // model is not tried again for this request.
      noteAttempt(candidate);
      lastError = (err as Error).message;
      continue;
    }
    if (!upstream.ok) {
      // A refusal (402, 429 or a 5xx) counted against the provider's budget. Fail over at once; two
      // refusals in a row mean the budget is spent or the outage has begun — but only once the
      // provider has made enough calls to plausibly be spent: a lone pair of adjacent scheduled fails
      // leaves it alive, and three in a row always kills it.
      state.calls += 1;
      state.failures += 1;
      state.streak += 1;
      if (state.streak >= DEAD_AFTER_FAILURES || (state.streak >= 2 && state.calls >= plannedCapacityMax(candidate.provider.id))) state.dead = true;
      noteAttempt(candidate);
      lastError = `the provider answered HTTP ${upstream.status}`;
      log(`${candidate.provider.id}/${candidate.model.id} refused HTTP ${upstream.status} (${state.failures} refusal${state.failures === 1 ? '' : 's'} this trial)`);
      try {
        await upstream.text();
      } catch {
        // The error body could not be read: the refusal stands either way.
      }
      continue;
    }
    state.calls += 1;
    state.streak = 0;
    log(`${candidate.provider.id}/${candidate.model.id} answered (rating ${candidate.rating})`);
    await respond(api, candidate, upstream, stream, res);
    return;
  }
  sendError(res, api, 503, api === 'anthropic' ? 'overloaded_error' : 'no_free_model', lastError);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    const body = await readJson(req);
    if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
    await serve('openai', body, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    const body = await readJson(req);
    if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
    await serve('anthropic', body, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
