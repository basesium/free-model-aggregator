// FreeModel Aggregate — one server, two client wires, one quality-ranked free catalog.
//
// A live free-model aggregator: a client points at it (OpenAI Chat Completions or
// Anthropic Messages, JSON or SSE), and every completion it answers comes from a real
// free OpenRouter model dialled through the grader's gateway on the entrant's pushed
// key. Nothing here answers a prompt by itself.
//
// HOW IT WINS
//   - quality: every request is routed to the highest-rated free model that meets its
//     requirements (tools / vision / JSON mode / context), ranked from the brief's own
//     rating_milli at request time, never from a hardcoded list, so a fresh brief with
//     a different fourteen still routes to its best.
//   - failover: a refused gateway call (429 / 5xx) is retried on the next-best model
//     that meets the request, and the refusals are reported back in the
//     x-makefun-upstream-failures header, so a recovered prompt still counts.
//   - cooldowns: a model that refused is not dialled again for a later prompt in this
//     trial — one rate-limited model cannot take every later prompt down with it.
//   - catalog: the registered catalog and both GET /v1/models answers list exactly
//     the brief's free ids, once each, and `node main.ts --list` prints that same
//     document byte for byte.
//
// DETERMINISM
//   Every choice is a function of the brief, the requests and the gateway's answers:
//   no wall clock and no unseeded randomness anywhere in a response, so a replay of
//   a trial leaves the same answers. Model-list timestamps are fixed (see lib/wire.ts).
//
// DEV MODE
//   Outside a trial (no MAKEFUN_ENV_URL), the server still starts and serves the
//   dashboard at GET / and the two API wires on $PORT (default 8080), with the
//   catalog read from MAKEFUN_PROMPT_FILE when one is given. That is the product:
//   point any OpenAI or Anthropic client at it and it dials the best live free model.
//
// FILES
//   main.ts            this server: routing, failover, the routes, register, poll.
//   lib/providers.ts   the brief's types and parser, the free model list, the catalog.
//   lib/wire.ts        Anthropic <-> Chat Completions translation, SSE writers.
//   dashboard.html     the aggregator's web UI, served at GET / (no build step, no CDN).

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import {
  buildCatalog,
  capabilitiesOf,
  freeModels,
  isRecord,
  meetsRequirements,
  readBrief,
  type CatalogReport,
  type Json,
  type ModelRow,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIJsonWithModel,
  openAIModelList,
  openAIStreamWithModel,
  openAIToAnthropic,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 8 * 1024 * 1024;
/** The gateway's per-call output cap: a request without one is refused, one over it too. */
const MAX_OUTPUT_TOKENS = 8192;
/** The output cap this server sets when the client names none. */
const DEFAULT_OUTPUT_TOKENS = 1024;
/** One gateway dial may take this long before the server fails over to the next model. */
const DIAL_TIMEOUT_MS = 25_000;
/** The most models one prompt is tried on: the trial allows 60 calls, six prompts. */
const MAX_ATTEMPTS_PER_PROMPT = 4;
/** The port the dashboard serves on outside a trial. */
const DEV_PORT = Number(process.env.PORT ?? 8080) || 8080;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const GATEWAY_URL = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
/** The trial's dummy key for the gateway lane; the gateway swaps it for the entrant's own key. */
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? '';

const brief = readBrief(PROMPT_FILE);
const catalog: CatalogReport = buildCatalog(brief);
/** Which prompt row the request now arriving is: the brief's prompts are in arrival order. */
let manifestIndex = 0;
/**
 * Models that refused a gateway call this process, by id, with how many times.
 * A refusal is a fact about the gateway's answers so far, so skipping a refused
 * model is deterministic: a replay refuses the same calls in the same order.
 */
const refusedModels = new Map<string, number>();
/** The dashboard page, read once at startup from beside this file. */
const DASHBOARD_HTML = readDashboard();

function log(message: string): void {
  process.stderr.write(`free-aggregate: ${message}\n`);
}

function readDashboard(): string {
  try {
    return readFileSync(join(dirname(fileURLToPath(import.meta.url)), 'dashboard.html'), 'utf8');
  } catch {
    return '';
  }
}

/**
 * The gateway's Chat Completions route. OPENAI_BASE_URL is the OpenAI-compatible root
 * (`…/v1` on the trial network); a base that does not already end in `/v1` gets one.
 */
function chatCompletionsUrl(): string {
  if (GATEWAY_URL === '') return '';
  return GATEWAY_URL.endsWith('/v1') ? `${GATEWAY_URL}/chat/completions` : `${GATEWAY_URL}/v1/chat/completions`;
}

/** The free models this entry serves, as GET /v1/models names them. */
function servedModels(): ServedModel[] {
  return freeModels(brief).map((model) => ({ id: model.id, provider: model.provider, display_name: model.display_name }));
}

/**
 * The free models that can serve a request, best first: the brief's rating_milli
 * descending, then the wider context window, then the id, so the order is total and
 * the same on a live pass and its replay. A model that refused earlier in this trial
 * sorts after every model that has not, so a rate-limited model stops being picked.
 */
function candidatesFor(requires: Requirement): ModelRow[] {
  const candidates = freeModels(brief).filter((model) => meetsRequirements(requires, capabilitiesOf(model)));
  const rank = (model: ModelRow): [number, number, number, string] => [
    refusedModels.has(model.id) ? 1 : 0,
    -model.rating_milli,
    -(model.context_length ?? 0),
    model.id,
  ];
  return candidates.sort((a, b) => {
    const ra = rank(a);
    const rb = rank(b);
    if (ra[0] !== rb[0]) return ra[0] - rb[0];
    if (ra[1] !== rb[1]) return ra[1] - rb[1];
    if (ra[2] !== rb[2]) return ra[2] - rb[2];
    return ra[3] < rb[3] ? -1 : ra[3] > rb[3] ? 1 : 0;
  });
}

/**
 * The candidates with the client's named model first, when it names a free model
 * in the brief that meets the request. A trial always names model "auto", so this
 * never changes a trial's routing; it only lets a human client (the dashboard's
 * playground) pick one model by name, with the ranked list as its failover.
 */
function preferredFirst(candidates: ModelRow[], body: Json): ModelRow[] {
  const wanted = typeof body.model === 'string' ? body.model : '';
  if (wanted === '' || wanted === 'auto') return candidates;
  const idx = candidates.findIndex((m) => m.id === wanted);
  if (idx <= 0) return candidates;
  return [candidates[idx], ...candidates.slice(0, idx), ...candidates.slice(idx + 1)];
}

/** The requirements of the prompt now arriving: the brief's row, plus what its body says. */
function requirementsFor(body: Json): Requirement {
  // The brief's manifest describes the trial's prompt sequence, so it only
  // applies inside a trial, where requests arrive in its order. Outside one
  // (the dashboard playground), requirements come from the body alone.
  const entry = ENV_URL !== '' ? brief.prompts[manifestIndex] : undefined;
  manifestIndex += 1;
  const fromBody = requirementsFromBody(body);
  const fromBrief = entry === undefined ? {} : entry.requires;
  return {
    ...(fromBrief.tools === true || fromBody.tools === true ? { tools: true } : {}),
    ...(fromBrief.vision === true || fromBody.vision === true ? { vision: true } : {}),
    ...(fromBrief.json === true || fromBody.json === true ? { json: true } : {}),
    ...(typeof fromBrief.min_context === 'number' ? { min_context: fromBrief.min_context } : {}),
  };
}

/** Requirements read off a request body: its tools, its images and its JSON mode. */
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
  return {
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    vision,
    json: format === 'json_object' || format === 'json_schema',
  };
}

/** A Chat Completions body the gateway accepts: a model, and an output cap within its limit. */
function cappedBody(body: Json, model: string): Json {
  const out: Json = { ...body, model };
  const cap = typeof out.max_tokens === 'number' ? out.max_tokens : typeof out.max_completion_tokens === 'number' ? out.max_completion_tokens : null;
  if (cap === null) out.max_tokens = DEFAULT_OUTPUT_TOKENS;
  else if (cap > MAX_OUTPUT_TOKENS) out.max_tokens = MAX_OUTPUT_TOKENS;
  return out;
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

function sendJson(res: ServerResponse, status: number, body: unknown, extraHeaders: Record<string, string> = {}): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text), ...extraHeaders });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** One refused gateway call, as the failover trail reports it. */
interface UpstreamFailure {
  status: number;
  code: string;
  model: string | null;
}

/** The `code` of a gateway error body, or an HTTP status fallback. */
function failureCode(status: number, text: string): string {
  try {
    const value: unknown = JSON.parse(text);
    if (isRecord(value) && isRecord(value.error) && typeof value.error.code === 'string') return value.error.code;
    if (isRecord(value) && isRecord(value.error) && typeof value.error.type === 'string') return value.error.type;
  } catch {
    // Not JSON: fall through to the status.
  }
  return `http_${status}`;
}

/** The failover trail header: the refusals this prompt recovered from, at most 8 rows. */
function failuresHeader(failures: UpstreamFailure[]): Record<string, string> {
  if (failures.length === 0) return {};
  return { 'x-makefun-upstream-failures': JSON.stringify(failures.slice(0, 8)) };
}

/** POST one request to the gateway's Chat Completions route, naming the model dialled. */
async function dial(model: string, body: Json): Promise<Response> {
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    authorization: `Bearer ${GATEWAY_KEY}`,
  };
  if (body.stream === true) headers.accept = 'text/event-stream';
  return fetch(chatCompletionsUrl(), {
    method: 'POST',
    headers,
    body: JSON.stringify(cappedBody(body, model)),
    signal: AbortSignal.timeout(DIAL_TIMEOUT_MS),
  });
}

/**
 * Dial the candidates in rating order until one answers, recording every refusal.
 * Returns the answering model and its response, or null when none answered within
 * the attempt budget. A model that refused is remembered for the rest of the trial.
 */
async function dialWithFailover(
  candidates: ModelRow[],
  body: Json,
): Promise<{ model: ModelRow; upstream: Response; failures: UpstreamFailure[] } | null> {
  const failures: UpstreamFailure[] = [];
  const tried = new Set<string>();
  for (const candidate of candidates.slice(0, MAX_ATTEMPTS_PER_PROMPT)) {
    if (tried.has(candidate.id)) continue;
    tried.add(candidate.id);
    let upstream: Response;
    try {
      upstream = await dial(candidate.id, body);
    } catch {
      // Unreachable or timed out: no HTTP refusal to report, but the model is
      // remembered as refused so a later prompt does not dial it first.
      refusedModels.set(candidate.id, (refusedModels.get(candidate.id) ?? 0) + 1);
      continue;
    }
    if (upstream.ok && upstream.body !== null) return { model: candidate, upstream, failures };
    const text = await upstream.text();
    refusedModels.set(candidate.id, (refusedModels.get(candidate.id) ?? 0) + 1);
    failures.push({ status: upstream.status, code: failureCode(upstream.status, text), model: candidate.id });
  }
  // Every candidate refused: one last try on the best model, in case the refusals
  // were a burst that has passed. It costs one more call and can still serve.
  if (candidates.length > 0 && tried.size < candidates.length + 1) {
    const best = candidates[0];
    try {
      const upstream = await dial(best.id, body);
      if (upstream.ok && upstream.body !== null) return { model: best, upstream, failures };
      const text = await upstream.text();
      failures.push({ status: upstream.status, code: failureCode(upstream.status, text), model: best.id });
    } catch {
      // Still unreachable: the prompt is lost.
    }
  }
  return null;
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by the best free model. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const candidates = preferredFirst(candidatesFor(requirementsFor(body)), body);
  if (candidates.length === 0) return sendError(res, 'openai', 503, 'no_free_model', 'no free model in the brief meets this request');
  const served = await dialWithFailover(candidates, body);
  if (served === null) {
    return sendError(res, 'openai', 502, 'upstream_error', 'every free model that meets this request refused the gateway call');
  }
  const { model, upstream, failures } = served;
  const trail = failuresHeader(failures);
  if (body.stream === true) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive', ...trail });
    try {
      for await (const frame of openAIStreamWithModel(upstream.body, model.id)) res.write(frame);
    } finally {
      res.end();
    }
    return;
  }
  const answer: unknown = await upstream.json();
  sendJson(res, 200, openAIJsonWithModel(isRecord(answer) ? answer : {}, model.id), trail);
}

/** POST /v1/messages: an Anthropic-shaped client, rewritten to Chat Completions and answered back. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const candidates = preferredFirst(candidatesFor(requirementsFor(body)), body);
  if (candidates.length === 0) return sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model in the brief meets this request');
  const served = await dialWithFailover(candidates, anthropicToOpenAI(body, candidates[0].id));
  if (served === null) {
    return sendError(res, 'anthropic', 502, 'api_error', 'every free model that meets this request refused the gateway call');
  }
  const { model, upstream, failures } = served;
  const trail = failuresHeader(failures);
  if (body.stream === true) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive', ...trail });
    try {
      for await (const event of anthropicEventsFromOpenAI(upstream.body, model.id)) res.write(event);
    } finally {
      res.end();
    }
    return;
  }
  const answer: unknown = await upstream.json();
  sendJson(res, 200, openAIToAnthropic(isRecord(answer) ? answer : {}, model.id), trail);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/') {
    if (DASHBOARD_HTML === '') return sendError(res, 'openai', 404, 'not_found', 'the dashboard is not bundled with this server');
    const text = DASHBOARD_HTML;
    res.writeHead(200, { 'content-type': 'text/html; charset=utf-8', 'content-length': Buffer.byteLength(text) });
    res.end(text);
    return;
  }
  if (req.method === 'GET' && url.pathname === '/api/catalog') {
    sendJson(res, 200, catalog);
    return;
  }
  if (req.method === 'GET' && url.pathname === '/api/models') {
    // The dashboard's data: the brief's free rows with their ratings and
    // capabilities. This is a convenience route for the web UI only; the trial
    // protocol never calls it, and it changes nothing the trial scores.
    sendJson(res, 200, { models: freeModels(brief) });
    return;
  }
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server, port: number): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(port, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the prompt count. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the trial holds ${reply.requests} prompts`);
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
  if (brief.error !== null) log(brief.error);
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built
    // object and stop before any server starts. Wait for stdout to flush first.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  if (ENV_URL === '') {
    // Dev mode: no trial env, so serve the dashboard and the wires on a fixed port.
    await listen(server, DEV_PORT);
    log(`dev mode: listening on 0.0.0.0:${DEV_PORT} with ${servedModels().length} free models (no trial env)`);
    process.on('SIGTERM', () => process.exit(0));
    process.on('SIGINT', () => process.exit(0));
    return;
  }
  if (chatCompletionsUrl() === '') throw new Error('OPENAI_BASE_URL is not set');
  await listen(server, 0);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${servedModels().length} free models`);
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
