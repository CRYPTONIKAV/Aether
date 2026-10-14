// Line-art icon set ported verbatim from the old panel — stroke-only SVGs
// that inherit currentColor, so every semantic text color just works.
const ICONS = {
  activity: <path d="M3 12h4l2.5-7 5 14 2.5-7H21"/>,
  attention: <><path d="M12 4 2 20h20L12 4z"/><path d="M12 10v4M12 17h.01"/></>,
  audit: <><rect x="4" y="3" width="16" height="18" rx="1.5"/><path d="M8 8h8M8 12h8M8 16h5"/></>,
  traces: <><circle cx="5" cy="5" r="2"/><circle cx="5" cy="12" r="2"/><circle cx="5" cy="19" r="2"/><path d="M9.5 5H19M9.5 12h6.5M9.5 19H19"/></>,
  policy: <path d="M12 3l7 3v6c0 4.5-3 7.5-7 9-4-1.5-7-4.5-7-9V6l7-3z"/>,
  apps: <><rect x="3.5" y="3.5" width="7" height="7" rx="1.5"/><rect x="13.5" y="3.5" width="7" height="7" rx="1.5"/><rect x="3.5" y="13.5" width="7" height="7" rx="1.5"/><rect x="13.5" y="13.5" width="7" height="7" rx="1.5"/></>,
  contacts: <><rect x="5" y="3" width="14" height="18" rx="2"/><circle cx="12" cy="9.5" r="2.6"/><path d="M8.5 16c0-1.9 1.6-3 3.5-3s3.5 1.1 3.5 3"/></>,
  tasks: <><path d="m4 12 4 4 12-12"/><path d="M4 6h9M4 18h9" strokeOpacity=".5"/></>,
  memory: <><circle cx="9" cy="8" r="3.2"/><path d="M4 20c0-3 2.2-5 5-5s5 2 5 5"/><circle cx="17" cy="7" r="2.2" strokeOpacity=".6"/><path d="M14.5 12.5c1.8.4 3 1.8 3.2 3.7" strokeOpacity=".6"/></>,
  settings: <><circle cx="12" cy="12" r="3"/><path d="M19.4 13.5a1.7 1.7 0 0 0 .34 1.87l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.7 1.7 0 0 0-1.87-.34 1.7 1.7 0 0 0-1.04 1.56V19.6a2 2 0 1 1-4 0v-.09a1.7 1.7 0 0 0-1.04-1.56 1.7 1.7 0 0 0-1.87.34l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.7 1.7 0 0 0 .34-1.87 1.7 1.7 0 0 0-1.56-1.04H2.4a2 2 0 1 1 0-4h.09a1.7 1.7 0 0 0 1.56-1.04 1.7 1.7 0 0 0-.34-1.87l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.7 1.7 0 0 0 1.87.34H8.5a1.7 1.7 0 0 0 1.04-1.56V2.4a2 2 0 1 1 4 0v.09c0 .68.4 1.29 1.04 1.56.62.26 1.34.14 1.87-.34l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.7 1.7 0 0 0-.34 1.87c.26.62.87 1.04 1.56 1.04h.09a2 2 0 1 1 0 4h-.09a1.7 1.7 0 0 0-1.56 1.04z"/></>,
  collapse: <path d="M4 6h16M4 12h10M4 18h16"/>,
  menu: <path d="M4 6h16M4 12h16M4 18h16"/>,
  bell: <><path d="M6 9a6 6 0 1 1 12 0c0 4 1.5 5.5 2 6H4c.5-.5 2-2 2-6Z"/><path d="M10 19a2 2 0 0 0 4 0"/></>,
  sun: <><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></>,
  moon: <path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8Z"/>,
  shield: <><path d="m12 3-8 3v5c0 5 3.4 8.3 8 10 4.6-1.7 8-5 8-10V6Z"/><path d="m9 12 2 2 4-5"/></>,
  arrow: <path d="m9 18 6-6-6-6"/>,
  github: <path d="M12 2a10 10 0 0 0-3.16 19.49c.5.09.68-.22.68-.48v-1.7c-2.78.6-3.37-1.34-3.37-1.34-.46-1.16-1.11-1.47-1.11-1.47-.91-.62.07-.6.07-.6 1 .07 1.53 1.03 1.53 1.03.9 1.53 2.36 1.09 2.93.83.09-.65.35-1.09.63-1.34-2.22-.25-4.56-1.11-4.56-4.94 0-1.09.39-1.99 1.03-2.69-.1-.25-.45-1.27.1-2.64 0 0 .84-.27 2.75 1.03a9.6 9.6 0 0 1 5 0c1.91-1.3 2.75-1.03 2.75-1.03.55 1.37.2 2.39.1 2.64.64.7 1.03 1.6 1.03 2.69 0 3.84-2.34 4.68-4.57 4.93.36.31.68.92.68 1.85v2.75c0 .26.18.58.69.48A10 10 0 0 0 12 2Z"/>,
  calendar: <><rect x="3" y="5" width="18" height="16" rx="2"/><path d="M7 3v4m10-4v4M3 10h18"/></>,
  folder: <path d="M3 6a1 1 0 0 1 1-1h5l2 2h9a1 1 0 0 1 1 1v10a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1Z"/>,
  mail: <><rect x="3" y="5" width="18" height="14" rx="2"/><path d="m3 7 9 6 9-6"/></>,
  discord: <><path d="M7 8c-2 .6-3 3-3 7.5 1.5 1.2 3 1.8 4.5 2l.7-1.3c-.7-.2-1.3-.5-1.9-.9.2-.1.3-.2.5-.3 3.2 1.4 6.8 1.4 10 0l.5.3c-.6.4-1.2.7-1.9.9l.7 1.3c1.5-.2 3-.8 4.5-2 0-4.5-1-7-3-7.5-.9-.3-1.9-.5-2.9-.6l-.4.9a13 13 0 0 0-4.8 0l-.4-.9c-1 .1-2 .3-2.9.6Z"/><circle cx="9.5" cy="13" r="1"/><circle cx="14.5" cy="13" r="1"/></>,
  clock: <><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></>,
  lock: <><rect x="4" y="10" width="16" height="11" rx="2"/><path d="M8 10V7a4 4 0 0 1 8 0v3"/></>,
  user: <><circle cx="12" cy="8" r="4"/><path d="M4 21a8 8 0 0 1 16 0"/></>,
};

export default function Icon({ name, className }) {
  const body = ICONS[name];
  if (!body) return null;
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6"
      className={className} aria-hidden="true">{body}</svg>
  );
}
