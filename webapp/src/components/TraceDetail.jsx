// The expanded trace: what triggered it, every gated call with its ruling,
// the reasoning, the reply, and the result. All plain text, colored only by
// the decision classes.
import { appLabel, personName, describeTool, describeRule, describeObservation, describeRecall, describeTrigger, humanize, plainText, summarizeResult, quote, paramRows, AUDIT_DECISIONS } from "../lib/describe.js";
import { fmtTime } from "../lib/format.js";
import { statusColor } from "../lib/status.js";

function CallRow({ c }) {
  return (
    <div data-testid="trace-call" className="mt-2.5 rounded-md border border-line bg-card px-3 py-2.5">
      <div className="flex items-baseline justify-between gap-3">
        {/* the recorded plain sentence (the model's own, or the backend's
            vocabulary) — scrubbed, since model text is untrusted; the raw
            name is only a fallback for rows nothing described */}
        <span className="text-sm text-ink">{c.plain ? plainText(c.plain) : describeTool(c.name)}</span>
        <span className={`shrink-0 text-xs font-medium ${statusColor(c.decision === "error" ? "error" : c.decision)}`}>
          {AUDIT_DECISIONS[c.decision] || humanize(c.decision)}
        </span>
      </div>
      {c.matched_rule && describeRule(c.matched_rule) ? (
        <div className="mt-1 text-xs text-ink-3">{describeRule(c.matched_rule)}</div>
      ) : null}
      {/* a parked call's result is an instruction to the model ("held for
          approval (#N) — …"), never human text — the Parked line says it */}
      {c.result && !c.approval_id ? (
        <div className={`mt-1 text-xs ${c.is_error ? "text-danger" : "text-ink-3"}`}>
          {c.is_error ? `failed: ${quote(plainText(c.result), 160)}` : summarizeResult(c.result)}
        </div>
      ) : null}
      {c.approval_id ? (
        <div className="mt-1 text-xs text-ink-3">
          Parked as <a href="#/attention" className="text-accent underline">approval #{c.approval_id}</a>
        </div>
      ) : null}
    </div>
  );
}

export default function TraceDetail({ t }) {
  const p = t.payload || {};

  return (
    <div>
      {t.kind === "turn" && p.trigger && (
        <>
          {(p.trigger.messages || []).map((m, i) => (
            <div data-testid="trace-line" key={`m${i}`} className="text-sm text-ink-2">
              <span className="text-ink-3">{appLabel(m.surface) || "chat"} · </span>
              {personName(m.surface, m.handle)} said: {quote(m.text, 220)}
            </div>
          ))}
          {(p.trigger.observations || []).map((o, i) => (
            <div data-testid="trace-line" key={`o${i}`} className="mt-1 text-xs text-ink-3">
              Also seen: {describeObservation(o)}
            </div>
          ))}
          {(p.trigger.recalled || []).map((r, i) => (
            <div data-testid="trace-line" key={`r${i}`} className="mt-1 text-xs text-ink-3">
              Kept in mind: {describeRecall(r)}
            </div>
          ))}
        </>
      )}

      {t.kind === "routine" && p.routine && (
        <>
          <div data-testid="trace-line" className="text-sm text-ink-2">
            Routine "{p.routine.label}" fired
            {p.routine.trigger ? <span className="text-ink-3"> — {describeTrigger(p.routine.trigger)}</span> : null}
          </div>
          {p.event && (
            <div data-testid="trace-line" className="mt-1 text-xs text-ink-3">
              Because: {describeObservation(p.event)}
            </div>
          )}
        </>
      )}

      {t.kind === "scheduled" && p.action && (
        <div data-testid="trace-line" className="text-sm text-ink-2">
          Scheduled action "{p.action.label}"
          <span className="text-ink-3"> — due {fmtTime(p.action.run_at)}</span>
        </div>
      )}

      {t.kind === "carry_out" && (
        <>
          <div data-testid="trace-line" className="text-sm text-ink-2">
            Carrying out approval #{p.approval_id}: {describeTool(p.tool)}
            {p.decided_by ? <span className="text-ink-3"> — decided by {p.decided_by === "user" ? "you" : appLabel(p.decided_by)}</span> : null}
          </div>
          {paramRows(p.params).length > 0 && (
            <dl className="mt-2 grid grid-cols-[max-content_1fr] gap-x-4 gap-y-1 text-sm">
              {paramRows(p.params).map(([k, v]) => (
                <div key={k} className="contents">
                  <dt className="text-ink-3">{k}</dt>
                  <dd className="text-ink-2">{v}</dd>
                </div>
              ))}
            </dl>
          )}
        </>
      )}

      {(p.calls || []).map((c, i) => <CallRow key={i} c={c} />)}

      {Array.isArray(p.reasoning) && p.reasoning.length > 0 && (
        <>
          <div className="mt-4 text-xs font-medium tracking-wide text-ink-3 uppercase">What it thought</div>
          {/* the loop can repeat a line across steps — collapse consecutive dupes */}
          {p.reasoning.filter((r, i) => i === 0 || r !== p.reasoning[i - 1]).map((r, i) => (
            <div key={i} className="mt-1 text-xs text-ink-2">{plainText(r)}</div>
          ))}
        </>
      )}

      {p.reply && (
        <>
          <div className="mt-4 text-xs font-medium tracking-wide text-ink-3 uppercase">What it answered</div>
          <div className="mt-1 text-sm text-ink-2">{plainText(p.reply)}</div>
        </>
      )}

      {p.result !== undefined && t.kind !== "turn" && (
        <div className={`mt-3 text-xs ${p.is_error ? "text-danger" : "text-ink-3"}`}>
          {p.is_error ? `failed: ${quote(plainText(String(p.result)), 200)}` : summarizeResult(String(p.result))}
        </div>
      )}

      {p.error && (
        <div className="mt-2 text-xs text-danger">error: {plainText(p.error)}</div>
      )}
    </div>
  );
}
