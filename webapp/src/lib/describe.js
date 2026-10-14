// The plain-language layer: no raw tool name, no raw rule id, no raw
// outcome string ever reaches the screen. Ported from the old panel's
// describe functions with the wording made professional — the structure
// and the plain-language guarantees are unchanged.
//
// personName needs the people list; the store bridges it in via
// setPeople() (same failure mode as before: names fall back to handles).

let people = [];

export function setPeople(list) {
  people = list || [];
}

export function personName(platform, handle) {
  if (!handle) return "someone";
  if (handle === "user") return "you";
  const match = people.find((p) => p.handles.some((hd) => hd.handle === handle && (!platform || hd.platform === platform)));
  return match ? match.display_name : handle;
}

export const ACTOR_LABELS = {
  agent: "Aether", scheduler: "Scheduled check", system: "Aether (automatic)", user: "You",
  owner: "You", routine: "Routine",
};

// last-resort rendering for any internal token with no explicit label —
// "poll:unread" → "Poll unread", never the raw string
export function humanize(text) {
  const words = String(text || "").replace(/[_:]+/g, " ").trim();
  return words ? words[0].toUpperCase() + words.slice(1) : "";
}

export const AUDIT_DECISIONS = {
  allow: "Allowed", approve: "Held for approval", require_approval: "Held for approval",
  deny: "Blocked", filtered: "Filtered out", info: "Noted",
  error: "Failed before the gate",
};

export const OUTCOME_LABELS = {
  "parked for approval": "Awaiting your approval",
  "executed after approval": "Completed after your approval",
  "could not run after approval": "Could not run after approval",
  "approval expired": "Expired without a decision",
  "changed from the panel": "You made this change from the panel",
  "stored encrypted — value never recorded": "Saved an app key (value not recorded)",
  "entity note added": "Saved a note about someone",
  "identities merged": "Recognised two accounts as one person",
  "new identity created": "Recorded a new person",
  "action scheduled": "Scheduled for later",
  "source poll": "Routine poll",
  "filtered by contact allowlist": "Sender is not on your contact list",
  "read-only tool, no side effects": "Read-only; nothing was changed",
  "internal tool, touches only Aether's own state": "Only touched Aether's own memory",
  "reaches an external system or is hard to undo": "Requires approval — it changes something outside Aether",
  "unknown tool — held for a human decision": "Unrecognized action — held for your decision",
  "changes who Aether listens to, what it may do, or what it's connected to — the user decides that with one tap":
    "Changes who Aether listens to or what it's connected to — requires your approval",
  "an unrecognized config path can never apply silently — held for the user to look at":
    "An unrecognized setting — held for your review",
};

export const TRACE_KINDS = {
  turn: "Conversation",
  routine: "Routine",
  scheduled: "Scheduled",
  carry_out: "Approved action",
};

// audit outcomes that close an approval's story — the "recently decided"
// strip is carved out of the already-polled audit feed, no extra endpoint
export const DECIDED_OUTCOME = /^(approved|denied) by \w+$/;
export const SETTLED_OUTCOMES = new Set([
  "executed after approval", "could not run after approval", "approval expired",
]);

export function describeRule(rule) {
  if (!rule) return "";
  if (BUILTIN_PLAIN[rule]) return BUILTIN_PLAIN[rule][0];
  if (rule === "config:poll_tools") return "Your regular polls";
  if (rule.startsWith("user:")) {
    const text = describeToolPattern(rule.slice(5));
    return `Your rule: ${text[0].toLowerCase()}${text.slice(1)}`;
  }
  if (rule.startsWith("routine:")) return "One of your routines";
  return "";
}

export function describeOutcome(outcome) {
  if (!outcome) return "";
  if (OUTCOME_LABELS[outcome]) return OUTCOME_LABELS[outcome];
  // "approved by user" is the only shape in practice — decided_by defaults to
  // "user" — but an older row could name a surface, so keep the fallback plain
  const decided = /^(approved|denied) by (\w+)$/.exec(outcome);
  if (decided) {
    const word = decided[1] === "approved" ? "Approved" : "Denied";
    return decided[2] === "user" ? `${word} by you` : `${word} by you via ${appLabel(decided[2])}`;
  }
  // policy.py's fallback reason for a user rule without a note
  const userRule = /^user rule ['"](.+)['"]$/.exec(outcome);
  if (userRule) {
    const text = describeToolPattern(userRule[1]);
    return `Your rule: ${text[0].toLowerCase()}${text.slice(1)}`;
  }
  // loop.py's audit outcome when one of your routines fires
  const routineFired = /^routine '(.+)' fired on ([^/]+)\/(.+)$/.exec(outcome);
  if (routineFired) {
    const { title } = describeEvent({ source: routineFired[2], kind: routineFired[3], payload: {} });
    return `Routine "${routineFired[1]}" fired — ${title[0].toLowerCase()}${title.slice(1)}`;
  }
  if (outcome.startsWith("handle linked")) return "Linked an account to someone you know";
  return outcome[0].toUpperCase() + outcome.slice(1);
}

const APP_NAMES = {
  gmail: "Email", mail: "Email", email: "Email", outlook: "Email",
  calendar: "Calendar", gcal: "Calendar",
  slack: "Slack", telegram: "Telegram", discord: "Discord", whatsapp: "WhatsApp", teams: "Teams",
  github: "GitHub", drive: "Files", filesystem: "Files", fs: "Files",
  phone: "Phone (SMS)", sms: "Phone (SMS)", call: "Phone call",
  memory: "Aether's memory",
};

const VERBS = [
  [/^(send|forward)$/, "Sending"], [/^reply$/, "Replying to"], [/^(post|publish)$/, "Posting"],
  [/^(delete|remove)$/, "Deleting"], [/^cancel$/, "Cancelling"], [/^(pay|transfer)$/, "Paying or transferring"],
  [/^invite$/, "Inviting people to"], [/^book$/, "Booking"], [/^create$/, "Creating"],
  [/^(update|edit|modify)$/, "Changing"], [/^(write|upload)$/, "Writing"], [/^(note|remember|save)$/, "Saving"],
  [/^(read|get|fetch|list|search|find|query|watch)$/, "Reading"],
];

const NOUNS = { email: "email", emails: "email", mail: "email", message: "messages", messages: "messages",
  msg: "messages", chat: "chats", chats: "chats", event: "events", events: "events",
  file: "files", files: "files", issue: "issues", issues: "issues", note: "notes", notes: "notes",
  channel: "channels", channels: "channels", contact: "contacts", contacts: "contacts" };

const DEFAULT_NOUN = { Email: "email", Calendar: "events", Slack: "messages", Telegram: "messages",
  Discord: "messages", WhatsApp: "messages", Teams: "messages", GitHub: "issues", Files: "files" };

const NATIVE_TOOLS = {
  memory_search: "Searching memory",
  note_entity: "Saving a note about someone",
  entity_merge: "Recognising two accounts as the same person",
  entity_resolve: "Linking an account to a person",
  get_pending_approvals: "Checking what's awaiting your approval",
  schedule_action: "Scheduling an action for later",
  request_screen_capture: "Requesting a screenshot",
  send_chat_message: "Messaging you",
  get_config: "Reading its configuration",
  set_config: "Changing its configuration",
  explain_decision: "Explaining a past decision",
  verify_integrity: "Verifying the audit chain",
  create_routine: "Creating a routine",
  list_routines: "Listing routines",
  set_routine_enabled: "Enabling or disabling a routine",
  delete_routine: "Deleting a routine",
};

const EXTRA_SOURCES = { screen: "Screen", web: "Web", user: "You", agent: "Aether" };

// action idioms the verb/noun tables can't phrase — keyed by the
// post-composio-shift action, lowercase ("who_am_i" from GMAIL_WHO_AM_I)
const ACTION_IDIOMS = {
  who_am_i: (appName) => `Checking the connected ${appName ? `${appName} ` : ""}account`,
};

export function appLabel(source) {
  if (!source) return "";
  const key = String(source).toLowerCase();
  return APP_NAMES[key] || EXTRA_SOURCES[key] || key[0].toUpperCase() + key.slice(1);
}

// a contact rule's platform — "*" is the platform-agnostic entry, never a
// bare asterisk on the screen
export function platformLabel(platform) {
  if (!platform || platform === "*") return "Any platform";
  return appLabel(platform);
}

export function describeTool(name) {
  if (!name) return "An action";
  if (NATIVE_TOOLS[name]) return NATIVE_TOOLS[name];
  if (name.startsWith("ingest:")) return `Receiving a message on ${appLabel(name.slice(7))}`;
  return describeName(name);
}

const PARAM_LABELS = {
  to: "To", cc: "Cc", recipient: "To", recipients: "To", chat: "Chat", channel: "Channel", handle: "To",
  chat_ref: "To",
  subject: "Subject", title: "Title", body: "Message", text: "Message", message: "Message", content: "Message",
  start: "When", when: "When", run_at: "When", end: "Until", date: "Date", location: "Where",
  attendees: "People", amount: "Amount", note: "Note", query: "Looking for",
};

export function friendlyValue(value) {
  if (value === null || value === undefined || value === "") return "";
  if (Array.isArray(value)) return value.map(friendlyValue).filter(Boolean).join(", ");
  if (typeof value === "object") return Object.values(value).map(friendlyValue).filter(Boolean).join(" · ");
  const text = String(value);
  if (/^\d{4}-\d{2}-\d{2}T/.test(text)) return new Date(text).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
  return text.length > 280 ? `${text.slice(0, 280)}…` : text;
}

export function paramRows(params) {
  return Object.entries(params || {})
    .filter(([k, v]) => !k.startsWith("_") && friendlyValue(v))
    .map(([k, v]) => [PARAM_LABELS[k.toLowerCase()] || k.replace(/_/g, " ").replace(/^./, (c) => c.toUpperCase()), friendlyValue(v)]);
}

export function quote(text, max = 200) {
  const clean = String(text).replace(/\s+/g, " ").trim();
  return `“${clean.length > max ? `${clean.slice(0, max)}…` : clean}”`;
}

// Collapse every balanced {...} span to an ellipsis — a JSON blob pasted
// into model text never renders. An unterminated "{" stays literal.
function collapseJson(text) {
  const s = String(text);
  let out = "";
  let i = 0;
  while (i < s.length) {
    if (s[i] === "{") {
      let depth = 0;
      let j = i;
      for (; j < s.length; j++) {
        if (s[j] === "{") depth++;
        else if (s[j] === "}") { depth--; if (depth === 0) break; }
      }
      if (depth === 0) { out += "…"; i = j + 1; continue; }
    }
    out += s[i];
    i++;
  }
  return out;
}

// Model-authored text (reasoning, replies, errors) can still mention a raw
// tool name or paste a JSON result — transcribe both before it renders.
export function plainText(text) {
  return collapseJson(text).replace(/\b[a-z][a-z0-9]*__[A-Za-z0-9_]+\b/g, (m) => {
    const plain = describeTool(m);
    return plain[0].toLowerCase() + plain.slice(1);
  });
}

export function summarizeResult(result) {
  if (!result) return "";
  let parsed = null;
  try { parsed = JSON.parse(result); } catch { /* plain text */ }
  if (Array.isArray(parsed)) return parsed.length ? `Found ${parsed.length} new item${parsed.length === 1 ? "" : "s"}` : "Nothing new";
  if (parsed && typeof parsed === "object") {
    // composio-style envelopes: unwrap {successful, data, error, log_id},
    // keeping any display_name seen along the way — and never fall through
    // to quoting the raw JSON
    let core = parsed;
    let display = "";
    for (let i = 0; i < 3 && core && typeof core === "object" && !Array.isArray(core); i++) {
      if (typeof core.display_name === "string" && core.display_name) display = core.display_name;
      const keys = Object.keys(core);
      if (!(core.data && typeof core.data === "object")) break;
      if (!keys.every((k) => ["successful", "data", "error", "log_id", "display_name"].includes(k))) break;
      core = core.data;
    }
    if (core && typeof core === "object" && !Array.isArray(core)) {
      const who = core.from || core.sender || core.organizer;
      const what = core.subject || core.title || core.summary || core.text;
      const bits = [who ? `from ${personName("", who)}` : "", what ? quote(what, 140) : ""].filter(Boolean);
      if (bits.length) return bits.join(" — ");
      const identity = display || core.email || core.name || "";
      if (identity) return identity;
    }
    return "The call returned data";
  }
  return quote(plainText(result), 140);
}

export function describeEvent(e) {
  const p = e.payload || {};
  const app = appLabel(e.source);
  if (e.kind === "chat_message") {
    // the handle lives under _sender (events.py SENDER_KEY), not at the top
    const handle = p.handle || (p._sender && p._sender.handle) || "";
    const platform = (p._sender && p._sender.platform) || e.source;
    const who = handle ? personName(platform, handle) : "someone";
    return { title: who === "you" ? `You wrote on ${app}` : `Chat with ${who} on ${app}`, detail: p.text ? quote(p.text) : "" };
  }
  // the agent's own acts — the payload carries the raw tool name and the
  // raw result text, so this branch never touches paramRows. `plain` is
  // written by the backend on every new row and backfilled at read time
  // for old ones; it is still model-adjacent text, so plainText scrubs it
  if (e.source === "agent" && p.tool) {
    if (p.plain) return { title: plainText(p.plain), detail: p.detail ? summarizeResult(String(p.detail)) : "" };
    const what = describeTool(p.tool);
    const prefix = e.kind === "action_failed" ? "Failed — " : e.kind === "action_denied" ? "Not run — you denied " : "";
    const title = `${prefix}${what[0].toLowerCase()}${what.slice(1)}`;
    return { title, detail: p.detail ? summarizeResult(String(p.detail)) : "" };
  }
  if (e.kind === "screen_capture") {
    const people = Array.isArray(p.people) && p.people.length ? ` · people: ${p.people.join(", ")}` : "";
    return { title: `Looked at your screen${p.app ? ` — ${p.app}` : ""}`, detail: `${p.summary || p.user_note || ""}${people}` };
  }
  if (e.kind === "shared_image") {
    // the handle lives under _sender like every user-authored event
    const handle = p.handle || (p._sender && p._sender.handle) || "";
    const platform = (p._sender && p._sender.platform) || e.source;
    const who = handle ? personName(platform, handle) : "someone";
    const people = Array.isArray(p.people) && p.people.length ? ` · people: ${p.people.join(", ")}` : "";
    const name = p.filename ? ` — ${p.filename}` : "";
    const detail = `${p.description || p.caption || ""}${people}`;
    return { title: who === "you" ? `You shared an image${name}` : `${who} shared an image${name}`, detail };
  }
  if (e.kind === "shared_doc") {
    const handle = p.handle || (p._sender && p._sender.handle) || "";
    const platform = (p._sender && p._sender.platform) || e.source;
    const who = handle ? personName(platform, handle) : "someone";
    const name = p.filename ? ` — ${p.filename}` : "";
    // file content is arbitrary text — plainText collapses any JSON blob a
    // config file might contain before it renders
    const detail = p.text ? quote(plainText(p.text), 200) : "No readable text was found in it";
    return { title: who === "you" ? `You shared a document${name}` : `${who} shared a document${name}`, detail };
  }
  if (e.kind.startsWith("poll:")) {
    const noun = DEFAULT_NOUN[app] || "updates";
    const detail = p.result !== undefined ? summarizeResult(p.result)
      : [p.from ? `from ${personName(e.source, p.from)}` : "", p.subject || p.title ? quote(p.subject || p.title, 140) : ""].filter(Boolean).join(" — ");
    const title = app === "Email" || app === "Calendar" ? `Checked your ${app.toLowerCase()}` : `Checked ${app} for new ${noun}`;
    return { title, detail };
  }
  const words = e.kind.replace(/[_:]+/g, " ").trim();
  const fields = paramRows(p).map(([k, v]) => `${k}: ${v}`).join(" · ");
  return { title: `${words[0].toUpperCase()}${words.slice(1)} on ${app}`, detail: fields };
}

// A trace observation carries the backend's raw log line —
// "2026-10-10 14:47 mail/poll:unread (salience 7): {"from": …}" — which is
// exactly the kind of internal text the panel never shows. Parse the event
// payload back out and describe it like a live event instead.
export function describeObservation(o) {
  let payload = {};
  const m = /\):\s*(\{.*\})\s*$/s.exec(o.line || "");
  if (m) { try { payload = JSON.parse(m[1]); } catch { payload = {}; } }
  const { title, detail } = describeEvent({ source: o.source, kind: o.kind, payload });
  return detail ? `${title} — ${detail}` : title;
}

// A recalled memory's line is the backend's own plain citation —
// "[related memory — OSY.pdf, page 3] …" or "[related memory — @vedant on
// telegram, 2026-09-30] …". Unwrap it into "where: what", so the where
// stays in front and nothing internal leaks through.
export function describeRecall(r) {
  const m = /^\[related memory — (.*)\]\s*(.*)$/s.exec(r.line || "");
  const where = m ? m[1] : "";
  const text = m ? m[2] : r.line || "";
  return where ? `${where}: ${quote(text, 220)}` : quote(text, 220);
}

// A routine's trigger is a condition dict ({source, kind, from, contains}) —
// render it as a plain when-clause, never the raw dict or kind token.
export function describeTrigger(trigger) {
  if (!trigger) return "";
  if (typeof trigger === "string") return humanize(trigger);
  const parts = [];
  if (trigger.source && trigger.source !== "*") parts.push(appLabel(trigger.source));
  if (trigger.kind && trigger.kind !== "*") {
    parts.push(
      trigger.kind.startsWith("poll:")
        ? `new ${humanize(trigger.kind.slice(5)).toLowerCase()}`
        : humanize(trigger.kind),
    );
  }
  if (trigger.from && trigger.from !== "*") parts.push(`from ${personName(trigger.source, trigger.from)}`);
  if (trigger.contains) parts.push(`mentioning ${quote(trigger.contains, 60)}`);
  return parts.length ? `When ${parts.join(" · ")}` : "On any event";
}

function cleanPattern(pattern) {
  return pattern.replace(/^\^|\$$/g, "").replace(/\\b/g, "").replace(/\\/g, "");
}

function describeName(name) {
  const wildcard = /[.*+?\[\]]/;
  let [app, action] = name.includes("__") ? name.split("__") : ["", name];
  // composio actions are UPPER_SNAKE with the toolkit as the first word —
  // composio__GMAIL_SEND_EMAIL reads as gmail, never as "composio" (mirrors
  // _action_words in src/aether/agent/tools.py)
  if (app === "composio" && action) {
    const [toolkit, ...rest] = action.toLowerCase().split("_");
    app = toolkit;
    action = rest.join("_");
  }
  const appName = app && !wildcard.test(app) ? (APP_NAMES[app.toLowerCase()] || app[0].toUpperCase() + app.slice(1)) : "";
  if (!action || wildcard.test(action) && !/[a-z]{3,}/i.test(action.replace(/[.*+?\[\]]/g, ""))) {
    return appName ? `Anything in ${appName}` : "Any action";
  }
  const idiom = ACTION_IDIOMS[action.toLowerCase()];
  if (idiom) return idiom(appName);
  const words = action.replace(/[.*+?\[\]]/g, " ").toLowerCase().split(/[_\s]+/).filter(Boolean);
  const verbHit = words.map((w) => VERBS.find(([re]) => re.test(w))).find(Boolean);
  const noun = words.map((w) => NOUNS[w]).find(Boolean) || (appName && DEFAULT_NOUN[appName]) || "";
  if (!verbHit) {
    const phrase = words.join(" ");
    return appName ? `${phrase[0].toUpperCase()}${phrase.slice(1)} in ${appName}` : `${phrase[0].toUpperCase()}${phrase.slice(1)}`;
  }
  const verb = verbHit[1];
  const object = noun || "anything";
  const redundant = (appName === "Email" && noun === "email") || (appName === "Files" && noun === "files");
  return appName && !redundant ? `${verb} ${object} on ${appName}` : `${verb} ${object}`;
}

export function describeToolPattern(pattern) {
  const cleaned = cleanPattern(pattern).replace(/^\((.*)\)$/, "$1");
  const group = /^([^()|]*)\(([^()]+)\)([^()|]*)$/.exec(cleaned);
  const parts = group ? group[2].split("|").map((alt) => group[1] + alt + group[3]) : cleaned.split("|");
  return parts.map((part) => describeName(part.replace(/[()]/g, "")))
    .map((text, i) => (i ? text[0].toLowerCase() + text.slice(1) : text)).join(" or ");
}

export const DECISION_LABELS = {
  allow: "Runs automatically",
  approve: "Requires approval",
  require_approval: "Requires approval",
  deny: "Never allowed",
};

export const BUILTIN_PLAIN = {
  "builtin:config-tune": ["Chat-driven settings change", "A settings change made in chat (agent behavior, memory importance or provider settings) — it applies immediately."],
  "builtin:config-security": ["A security-sensitive settings change", "A change to your contacts, authorization, messaging or app servers — it always waits for your approval."],
  "builtin:internal": ["Aether's own records", "Remembering people and details, setting reminders, and requesting a screenshot only touch Aether's own memory."],
  "builtin:risky": ["Anything that reaches the outside world", "Sending, replying, posting, deleting, cancelling, paying, booking, inviting or creating something always waits for your approval."],
  "builtin:read-only": ["Read-only actions", "Reading, searching and listing your mail, chats and calendar — nothing is changed."],
  "builtin:walk-connect": ["Connecting an app from the /apps walk", "You completed the connect walk for this app in chat — the walk itself serves as your approval."],
  "builtin:oauth-consent": ["Connecting an app by signing in", "The sign-in you completed on the provider's own screen serves as your approval."],
  "default:fail-safe": ["Unrecognized actions", "An action Aether does not recognize is held for your approval before it runs."],
};
