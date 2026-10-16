// App shell: sidebar + topbar + the routed page. Owns only navigation chrome
// (collapse state, mobile drawer) — every server-backed list lives in the
// DataProvider so polls keep running and caches survive page switches.
import { useEffect, useState } from "react";
import Sidebar from "./components/Sidebar.jsx";
import Topbar from "./components/Topbar.jsx";
import { useData } from "./store.jsx";
import { useHashRoute } from "./hooks/useHashRoute.js";
import { hasToken } from "./lib/api.js";
import SettingsPage from "./pages/Settings.jsx";
import AttentionPage from "./pages/Attention.jsx";
import ActivityPage from "./pages/Activity.jsx";
import TracesPage from "./pages/Traces.jsx";
import AuditPage from "./pages/Audit.jsx";
import TasksPage from "./pages/Tasks.jsx";
import MemoryPage from "./pages/Memory.jsx";
import PolicyPage from "./pages/Policy.jsx";
import AppsPage from "./pages/Apps.jsx";
import ContactsPage from "./pages/Contacts.jsx";

const COLLAPSE_KEY = "aether_sidebar_collapsed";

const PAGES = {
  activity: { title: "Live Activity", el: ActivityPage },
  attention: { title: "Attention Required", el: AttentionPage },
  tasks: { title: "Tasks", el: TasksPage },
  memory: { title: "Memory & Context", el: MemoryPage },
  audit: { title: "Audit Log", el: AuditPage },
  traces: { title: "Decision Traces", el: TracesPage },
  policy: { title: "Authorization & Policies", el: PolicyPage },
  apps: { title: "Apps & Permissions", el: AppsPage },
  contacts: { title: "Contacts", el: ContactsPage },
  settings: { title: "Settings", el: SettingsPage },
};

export default function App() {
  const { approvals } = useData();
  const route = useHashRoute(Object.keys(PAGES));
  const [collapsed, setCollapsed] = useState(() => {
    try { return localStorage.getItem(COLLAPSE_KEY) === "1"; } catch { return false; }
  });
  const [navOpen, setNavOpen] = useState(false);

  const { title, el: Page } = PAGES[route];

  // no token and no destination — start at settings, same as before
  useEffect(() => {
    if (!hasToken() && !location.hash) location.hash = "#/settings";
  }, []);

  useEffect(() => {
    document.title = `Aether — ${title}`;
  }, [title]);

  // Escape closes the drawer; widening past the mobile breakpoint closes it
  useEffect(() => {
    const onKey = (e) => { if (e.key === "Escape") setNavOpen(false); };
    const mq = window.matchMedia("(min-width: 901px)");
    const onMq = (e) => { if (e.matches) setNavOpen(false); };
    document.addEventListener("keydown", onKey);
    mq.addEventListener("change", onMq);
    return () => {
      document.removeEventListener("keydown", onKey);
      mq.removeEventListener("change", onMq);
    };
  }, []);

  const toggleCollapse = () => setCollapsed((c) => {
    const next = !c;
    try { localStorage.setItem(COLLAPSE_KEY, next ? "1" : "0"); } catch { /* private mode */ }
    return next;
  });

  const attentionCount = approvals.length;

  return (
    <div className="flex min-h-screen">
      {navOpen && (
        <div
          className="fixed inset-0 z-30 bg-black/40 min-[901px]:hidden"
          onClick={() => setNavOpen(false)}
          aria-hidden="true"
        />
      )}
      <Sidebar
        page={route}
        collapsed={collapsed}
        navOpen={navOpen}
        attentionCount={attentionCount}
        onCollapse={toggleCollapse}
      />
      <div className="flex min-w-0 flex-1 flex-col">
        <Topbar title={title} attentionCount={attentionCount} onMenu={() => setNavOpen(true)} />
        <main data-testid="page" className="mx-auto w-full max-w-5xl flex-1 px-4 py-6 min-[901px]:px-6">
          <Page />
        </main>
      </div>
    </div>
  );
}
