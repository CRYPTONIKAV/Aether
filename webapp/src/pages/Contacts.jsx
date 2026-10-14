// Who may reach Aether: one card per connected app, each with its own
// allowlist and its own switch, plus the "Any platform" list that counts
// everywhere. The same lists chat's /config edits — the owner's
// authenticated taps are the approval; every change still lands through
// the same validated, persisted path and in the audit log.
import { useEffect, useState } from "react";
import PageHead from "../components/PageHead.jsx";
import { Button, Card, Field, Select } from "../components/ui.jsx";
import EmptyState, { NoToken } from "../components/EmptyState.jsx";
import { useToast } from "../components/Toasts.jsx";
import { useData } from "../store.jsx";
import { api, hasToken } from "../lib/api.js";
import { personName, platformLabel } from "../lib/describe.js";

const PLATFORMS = [
  ["*", "Any platform"],
  ["telegram", "Telegram"],
  ["discord", "Discord"],
  ["slack", "Slack"],
  ["whatsapp", "WhatsApp"],
  ["phone", "Phone (SMS)"],
  ["email", "Email"],
];

// an app card's own words: what its list governs and what its add box takes
const APP_META = {
  gmail: {
    hint: "While filtering is on, Aether only reads mail from these people — plus mail you sent them, and mail you sent yourself. Everything else is dropped before Aether ever sees it.",
    placeholder: "an email address",
  },
  telegram: { hint: "Only these people's Telegram messages reach Aether while filtering is on.", placeholder: "@name" },
  discord: { hint: "Only these people's Discord messages reach Aether while filtering is on.", placeholder: "@name" },
  slack: { hint: "Only these people's Slack messages reach Aether while filtering is on.", placeholder: "@name" },
};

const MODE_TEXT = {
  enforce: "Only the people on this list can reach Aether — everyone else is dropped before anything is stored.",
  off: "Everyone can reach Aether right now. The list is kept for when filtering comes back on.",
};

const appTitle = (key) => platformLabel(key === "gmail" ? "email" : key);

// One app's card: its own switch (which can follow the default), its own
// list with Remove, and an inline add — the card fixes the platform, so
// only the handle is typed.
function AppCard({ app, defaultMode, busy, write, jsonOpts }) {
  const toast = useToast();
  const [handle, setHandle] = useState("");
  const meta = APP_META[app.key] || { hint: "", placeholder: "a handle" };
  const mode = app.mode ?? defaultMode; // null mode = follows the default
  const own = app.mode !== null;

  const add = () => {
    const who = handle.trim();
    if (!who) { toast("Enter the person's handle first.", "err"); return; }
    write(
      `/api/contacts/apps/${app.key}/allowlist`,
      jsonOpts("POST", { platform: app.key, handle: who }),
      `${personName(app.key, who)} can now reach Aether on ${appTitle(app.key)}.`,
    );
    setHandle("");
  };

  const remove = (entry) => {
    // "any" entries live in the cross-app bucket — removing one takes it
    // off everywhere, exactly as the "Any platform" card's Remove does
    const path = entry.list === "any" ? "/api/contacts/allowlist" : `/api/contacts/apps/${app.key}/allowlist`;
    write(
      path,
      jsonOpts("DELETE", { platform: entry.platform, handle: entry.handle }),
      `${personName(entry.platform, entry.handle)} can no longer reach Aether.`,
    );
  };

  const setMode = (value) => {
    if (value === "default") {
      write(
        `/api/contacts/apps/${app.key}/mode`,
        jsonOpts("PUT", { mode: "default" }),
        `${appTitle(app.key)} follows the default again.`,
      );
      return;
    }
    write(
      `/api/contacts/apps/${app.key}/mode`,
      jsonOpts("PUT", { mode: value }),
      value === "enforce"
        ? `Only the people on the ${appTitle(app.key)} list can reach Aether now.`
        : `Everyone can reach Aether on ${appTitle(app.key)} now.`,
    );
  };

  return (
    <Card className="mb-4 p-4" data-testid={`app-card-${app.key}`}>
      <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
        <h2 className="text-sm font-semibold text-ink">
          {appTitle(app.key)}
          {app.connected ? null : <span className="ml-2 text-xs font-normal text-ink-3">not connected</span>}
        </h2>
        <Select
          data-testid={`app-mode-${app.key}`}
          value={own ? app.mode : "default"}
          disabled={busy}
          onChange={(e) => setMode(e.target.value)}
        >
          <option value="default">Default ({mode === "enforce" ? "only the list" : "everyone"})</option>
          <option value="enforce">Only this app's list</option>
          <option value="off">Everyone</option>
        </Select>
      </div>
      <p className="mb-3 text-xs text-ink-2">{meta.hint}</p>
      {app.allowlist.length === 0 ? (
        <EmptyState
          title="No one is on this app's list yet"
          sub={mode === "enforce"
            ? "While filtering is on, nobody at all can reach Aether here."
            : "Add people below so the list is ready when filtering comes on."}
        />
      ) : (
        <ul className="divide-y divide-line">
          {app.allowlist.map((entry) => (
            <li
              key={`${entry.platform}/${entry.handle}`}
              data-testid={`contact-row-${app.key}`}
              className="flex items-center justify-between gap-3 py-2.5"
            >
              <div className="min-w-0">
                <div className="truncate text-sm text-ink">
                  {personName(entry.platform, entry.handle)}
                  {entry.list === "any" ? <span className="ml-2 text-xs text-ink-3">any platform</span> : null}
                </div>
                <div className="text-xs text-ink-3">{entry.handle}</div>
              </div>
              <Button data-testid="contact-remove" disabled={busy} onClick={() => remove(entry)}>
                Remove
              </Button>
            </li>
          ))}
        </ul>
      )}
      <div className="mt-3 flex flex-wrap gap-2">
        <Field
          data-testid={`contact-handle-${app.key}`}
          type="text"
          placeholder={meta.placeholder}
          value={handle}
          disabled={busy}
          onChange={(e) => setHandle(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter") add(); }}
          className="min-w-0 flex-1"
        />
        <Button kind="primary" data-testid={`contact-add-${app.key}`} disabled={busy} onClick={add}>
          Add
        </Button>
      </div>
    </Card>
  );
}

export default function ContactsPage() {
  const toast = useToast();
  const { ensurePeople } = useData();
  const [state, setState] = useState(null); // {mode, allowlist, apps}
  const [error, setError] = useState("");
  const [platform, setPlatform] = useState("*");
  const [handle, setHandle] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (!hasToken()) return;
    ensurePeople(); // names win over handles wherever Aether knows the person
    let alive = true;
    api("/api/contacts")
      .then((d) => alive && setState(d))
      .catch((e) => alive && setError(e.message));
    return () => { alive = false; };
  }, []);

  // every write answers with the fresh state, so the page updates in one
  // round-trip — refusals arrive as plain words and only toast
  const write = async (path, opts, okMsg) => {
    setBusy(true);
    try {
      setState(await api(path, opts));
      if (okMsg) toast(okMsg, "ok");
    } catch (err) {
      toast(err.message, "err");
    } finally {
      setBusy(false);
    }
  };

  const jsonOpts = (method, body) => ({
    method,
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });

  const add = () => {
    const who = handle.trim();
    if (!who) { toast("Enter the person's handle first.", "err"); return; }
    write(
      "/api/contacts/allowlist",
      jsonOpts("POST", { platform, handle: who }),
      `${personName(platform, who)} can now reach Aether.`,
    );
    setHandle("");
  };

  const remove = (entry) => {
    write(
      "/api/contacts/allowlist",
      jsonOpts("DELETE", entry),
      `${personName(entry.platform, entry.handle)} can no longer reach Aether.`,
    );
  };

  const setMode = (mode) => {
    write(
      "/api/contacts/mode",
      jsonOpts("PUT", { mode }),
      mode === "enforce"
        ? "Only the people on the list can reach Aether now."
        : "Everyone can reach Aether now.",
    );
  };

  return (
    <div className="max-w-3xl">
      <PageHead
        eyebrow="Privacy"
        title="Contacts"
        sub="Who Aether sees and remembers messages from — one list per app, plus one that counts everywhere. The same lists chat's /config edits; changes apply to the next message."
      />
      {!hasToken() ? (
        <Card><NoToken /></Card>
      ) : error ? (
        <Card><EmptyState title="The contact list isn't available." sub={error} /></Card>
      ) : !state ? (
        <div className="py-2 text-sm text-ink-3">Loading…</div>
      ) : (
        <>
          {(state.apps || []).map((app) => (
            <AppCard
              key={app.key}
              app={app}
              defaultMode={state.mode}
              busy={busy}
              write={write}
              jsonOpts={jsonOpts}
            />
          ))}

          <Card className="mb-4 p-4">
            <h2 className="mb-2 text-sm font-semibold text-ink">The default, for every app without its own switch</h2>
            <p className="mb-3 text-xs text-ink-2" data-testid="contacts-mode-text">
              {MODE_TEXT[state.mode]}
            </p>
            <Select
              data-testid="contacts-mode"
              value={state.mode}
              disabled={busy}
              onChange={(e) => setMode(e.target.value)}
            >
              <option value="enforce">Only the people on the list</option>
              <option value="off">Everyone</option>
            </Select>
          </Card>

          <Card className="mb-4 p-4">
            <h2 className="mb-2 text-sm font-semibold text-ink">Any platform</h2>
            <p className="mb-3 text-xs text-ink-2">
              These people count on every app — Gmail, Discord, Telegram, Slack —
              wherever they match.
            </p>
            {state.allowlist.length === 0 ? (
              <EmptyState
                title="No one is on the list yet"
                sub={state.mode === "enforce"
                  ? "While filtering is on, nobody at all can reach Aether."
                  : "Add people below so the list is ready when filtering comes on."}
              />
            ) : (
              <ul className="divide-y divide-line">
                {state.allowlist.map((entry) => (
                  <li
                    key={`${entry.platform}/${entry.handle}`}
                    data-testid="contact-row"
                    className="flex items-center justify-between gap-3 py-2.5"
                  >
                    <div className="min-w-0">
                      <div className="truncate text-sm text-ink">
                        {personName(entry.platform, entry.handle)}
                      </div>
                      <div className="text-xs text-ink-3">
                        {platformLabel(entry.platform)} · {entry.handle}
                      </div>
                    </div>
                    <Button
                      data-testid="contact-remove"
                      disabled={busy}
                      onClick={() => remove(entry)}
                    >
                      Remove
                    </Button>
                  </li>
                ))}
              </ul>
            )}
          </Card>

          <Card className="p-4">
            <h2 className="mb-2 text-sm font-semibold text-ink">Add someone on any platform</h2>
            <p className="mb-3 text-xs text-ink-2">
              Handles match the way they arrive: @name for chat apps, a phone
              number for SMS or WhatsApp, an email address for mail. "Any
              platform" matches the same handle everywhere.
            </p>
            <div className="flex flex-wrap gap-2">
              <Select
                data-testid="contact-platform"
                value={platform}
                disabled={busy}
                onChange={(e) => setPlatform(e.target.value)}
              >
                {PLATFORMS.map(([value, label]) => (
                  <option key={value} value={value}>{label}</option>
                ))}
              </Select>
              <Field
                data-testid="contact-handle"
                type="text"
                placeholder="e.g. @friend or +91…"
                value={handle}
                disabled={busy}
                onChange={(e) => setHandle(e.target.value)}
                onKeyDown={(e) => { if (e.key === "Enter") add(); }}
                className="min-w-0 flex-1"
              />
              <Button kind="primary" data-testid="contact-add" disabled={busy} onClick={add}>
                Add
              </Button>
            </div>
          </Card>
        </>
      )}
    </div>
  );
}
