import Icon from "./Icon.jsx";
import { assetUrl, ASSET_V } from "../lib/assets.js";

// The routes in their original order. The attention count is a plain
// number — no badge, no dot, no fill.
const NAV_ITEMS = [
  ["activity", "Live Activity"],
  ["attention", "Attention Required"],
  ["tasks", "Tasks"],
  ["memory", "Memory & Context"],
  ["audit", "Audit Log"],
  ["traces", "Decision Traces"],
  ["policy", "Authorization & Policies"],
  ["apps", "Apps & Permissions"],
  ["contacts", "Contacts"],
  ["settings", "Settings"],
];

export default function Sidebar({ page, collapsed, navOpen, attentionCount, onCollapse }) {
  return (
    <aside
      className={`fixed inset-y-0 left-0 z-40 w-64 flex-col border-r border-line bg-surface
        min-[901px]:sticky min-[901px]:top-0 min-[901px]:h-screen min-[901px]:flex
        ${collapsed ? "min-[901px]:w-14" : "min-[901px]:w-60"} ${navOpen ? "flex" : "hidden"}`}
      aria-label="Primary"
    >
      <div className={`flex items-center gap-2.5 px-3.5 ${collapsed ? "min-[901px]:justify-center min-[901px]:px-0" : ""} py-4`}>
        <img
          src={assetUrl(`aether-mark.jpg?v=${ASSET_V}`)}
          alt="Aether"
          className="h-9 w-9 rounded-md object-cover"
        />
        {!collapsed && (
          <div className="min-[901px]:block">
            <div className="text-sm font-semibold tracking-wide text-ink">AETHER</div>
            <div className="text-xs text-ink-3">private agent</div>
          </div>
        )}
      </div>

      <div className={`px-3.5 pb-1 text-[11px] font-medium tracking-wide text-ink-3 uppercase ${collapsed ? "min-[901px]:hidden" : ""}`}>
        Mission control
      </div>

      <nav className="flex-1 overflow-y-auto px-2">
        {NAV_ITEMS.map(([id, label]) => {
          const active = page === id;
          return (
            <a
              key={id}
              href={`#/${id}`}
              aria-current={active ? "page" : undefined}
              className={`flex items-center gap-2.5 rounded-md px-2.5 py-2 text-sm
                ${collapsed ? "min-[901px]:justify-center min-[901px]:px-0" : ""}
                ${active ? "bg-subtle font-medium text-ink" : "text-ink-2 hover:bg-subtle hover:text-ink"}`}
            >
              <Icon name={id} className="h-4.5 w-4.5 shrink-0" />
              <span className={`min-w-0 truncate ${collapsed ? "min-[901px]:hidden" : ""}`}>{label}</span>
              {id === "attention" && attentionCount > 0 && (
                <span className={`ml-auto pl-1 text-sm font-medium text-warning ${collapsed ? "min-[901px]:hidden" : ""}`}>
                  {attentionCount}
                </span>
              )}
            </a>
          );
        })}
      </nav>

      <div className="border-t border-line p-2">
        <button
          type="button"
          onClick={onCollapse}
          aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
          title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
          className="hidden min-[901px]:flex min-[901px]:w-full min-[901px]:items-center min-[901px]:justify-center rounded-md px-2.5 py-2 text-ink-3 hover:bg-subtle hover:text-ink"
        >
          <Icon name="collapse" className="h-4.5 w-4.5" />
        </button>
      </div>
    </aside>
  );
}
