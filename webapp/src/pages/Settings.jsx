// Local-configuration page. The token lives only in this browser; saving or
// clearing it reloads the page so every poller restarts with the new state.
import { useEffect, useState } from "react";
import PageHead from "../components/PageHead.jsx";
import { Card, Field, Button } from "../components/ui.jsx";
import EmptyState from "../components/EmptyState.jsx";
import { useToast } from "../components/Toasts.jsx";
import { api, token, hasToken } from "../lib/api.js";
import { friendlyValue } from "../lib/describe.js";

export default function SettingsPage() {
  const toast = useToast();
  const [tokenValue, setTokenValue] = useState(() => token());
  const [personality, setPersonality] = useState("");
  const [personalityStatus, setPersonalityStatus] = useState({ text: "", bad: false });
  const [config, setConfig] = useState(null);
  const [configError, setConfigError] = useState("");

  useEffect(() => {
    if (!hasToken()) return;
    api("/api/settings/personality")
      .then((d) => setPersonality(d.text))
      .catch((err) => setPersonalityStatus({ text: `Could not load: ${err.message}`, bad: true }));
    api("/api/config")
      .then(setConfig)
      .catch((err) => setConfigError(err.message));
  }, []);

  const saveToken = () => {
    const val = tokenValue.trim();
    if (!val) { toast("Enter a token first.", "err"); return; }
    try { localStorage.setItem("aether_token", val); } catch { /* private mode */ }
    location.reload();
  };

  const clearToken = () => {
    try { localStorage.removeItem("aether_token"); } catch { /* private mode */ }
    location.reload();
  };

  const savePersonality = async () => {
    setPersonalityStatus({ text: "Saving…", bad: false });
    try {
      await api("/api/settings/personality", {
        method: "PUT", headers: { "content-type": "application/json" },
        body: JSON.stringify({ text: personality }),
      });
      setPersonalityStatus({ text: "Saved — it takes effect on Aether's next reply.", bad: false });
    } catch (err) {
      setPersonalityStatus({ text: `Not saved: ${err.message}`, bad: true });
    }
  };

  const s = (config && config.sections) || {};
  const messagingOn = ["telegram", "discord", "slack"]
    .filter((p) => s.messaging && s.messaging[p] && s.messaging[p].enabled);
  const servers = (s.mcp_servers || []).filter((x) => x.enabled);
  const summary = [
    ["Messaging", messagingOn.length ? messagingOn.join(", ") : "all off"],
    ["App servers (MCP)", servers.length ? servers.map((x) => x.name).join(", ") : "none"],
    ["Contact allowlist", (s.contacts && s.contacts.mode) || "off"],
    ["Approval window", s.authz ? `${s.authz.approval_ttl_hours} hours` : "—"],
    ["Quiet hours", (s.agent && s.agent.quiet_hours) || "none"],
    ["Provider", s.llm ? `${s.llm.provider} · ${s.llm.model}` : "—"],
  ];

  return (
    <div className="max-w-3xl">
      <PageHead
        eyebrow="Local configuration"
        title="Settings"
        sub="Configuration Aether actually persists and uses."
      />

      <Card className="mb-4 p-4">
        <h2 className="mb-2 text-sm font-semibold text-ink">API token</h2>
        <p className="mb-3 text-xs text-ink-2">
          Required to reach the chat, approvals, activity, audit and policy endpoints.
          Stored only in this browser's local storage.
        </p>
        <div className="flex gap-2">
          <Field
            type="password"
            placeholder="API token"
            value={tokenValue}
            onChange={(e) => setTokenValue(e.target.value)}
            className="min-w-0 flex-1"
          />
          <Button kind="primary" onClick={saveToken}>Save</Button>
          <Button onClick={clearToken}>Clear</Button>
        </div>
      </Card>

      <Card className="mb-4 p-4">
        <h2 className="mb-2 text-sm font-semibold text-ink">Personality</h2>
        <p className="mb-3 text-xs text-ink-2">
          Free text appended to Aether's real system prompt on every turn — this changes how it
          behaves, not just how it looks. Leave empty for the default behavior.
        </p>
        <textarea
          data-testid="personality-input"
          rows={5}
          disabled={!hasToken()}
          placeholder={hasToken() ? "e.g. Be terse and a little dry. Never use emoji." : "Set your API token above first"}
          value={personality}
          onChange={(e) => setPersonality(e.target.value)}
          className="w-full resize-y rounded-md border border-line-strong bg-card px-3 py-2 font-mono text-sm text-ink placeholder:text-ink-3 disabled:opacity-50"
        />
        <div className="mt-3 flex items-center gap-3">
          <Button kind="primary" data-testid="personality-save" disabled={!hasToken()} onClick={savePersonality}>
            Save personality
          </Button>
          <span
            data-testid="personality-status"
            className={`text-xs ${personalityStatus.bad ? "text-danger" : "text-ink-3"}`}
          >
            {personalityStatus.text}
          </span>
        </div>
      </Card>

      <Card className="p-4">
        <h2 className="mb-2 text-sm font-semibold text-ink">Configuration</h2>
        <p className="mb-3 text-xs text-ink-2">
          What Aether is running with — config.yaml merged with changes made in chat. Read-only
          here; change it from chat with /config, under the same approval gate as everything else.
          The one exception: the contact list, which the{" "}
          <a href="#/contacts" className="text-accent underline">Contacts</a> page edits directly.
        </p>
        {hasToken() ? (
          configError ? (
            <EmptyState title="Configuration isn't available." sub={configError} />
          ) : config ? (
            <>
              <dl className="grid grid-cols-[max-content_1fr] gap-x-4 gap-y-1.5 text-sm">
                {summary.map(([k, v]) => (
                  <div key={k} className="contents">
                    <dt className="text-ink-3">{k}</dt>
                    <dd className="text-ink-2">{String(v)}</dd>
                  </div>
                ))}
              </dl>
              {config.overrides && config.overrides.length > 0 && (
                <>
                  <div className="mt-4 mb-1.5 text-xs font-medium tracking-wide text-ink-3 uppercase">
                    Changed from chat
                  </div>
                  {config.overrides.map((o) => (
                    <div key={o.path} className="font-mono text-xs text-ink-2">
                      {o.path} = {friendlyValue(o.value)}
                    </div>
                  ))}
                </>
              )}
            </>
          ) : (
            <div className="py-2 text-sm text-ink-3">Loading…</div>
          )
        ) : (
          <div className="py-2 text-sm text-ink-3">Set your API token above first.</div>
        )}
      </Card>
    </div>
  );
}
