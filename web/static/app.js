"use strict";

const state = {
  providers: [],
  service: { host: "127.0.0.1", port: 4444, base_url: "http://127.0.0.1:4444" },
  activeProvider: null,
  busyActions: new Set(),
  refreshing: false,
  generating: false,
};

const byId = (id) => document.getElementById(id);
const formatTime = (value) => {
  if (!value) return "Never";
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? String(value) : parsed.toLocaleString();
};
const statusClass = (status) => String(status || "OFFLINE").toLowerCase().replaceAll("_", "-");
const prettyStatus = (status) => String(status || "OFFLINE").toUpperCase();
const displayAuth = (auth) => auth === "ok" ? "OK" : auth === "required" ? "AUTH_REQUIRED" : "UNKNOWN";
const providerGlyph = (name) => String(name || "P").split(/[\s-]+/).map((part) => part[0]).join("").slice(0, 2).toUpperCase();

async function apiFetch(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  let payload = null;
  try { payload = await response.json(); } catch { payload = null; }
  if (!response.ok) {
    const detail = payload?.detail;
    const code = detail?.error || payload?.error || `HTTP_${response.status}`;
    const message = detail?.message || detail || payload?.message || response.statusText || "Request failed";
    throw new Error(`${code}: ${message}`);
  }
  return payload;
}

function node(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}

function statusBadge(status) {
  return node("span", `status-badge ${statusClass(status)}`, prettyStatus(status));
}

function keepaliveLabel(provider) {
  const policy = provider.keepalive || {};
  if (!policy.enabled) return "Disabled";
  return `Daily ${String(policy.hour ?? 0).padStart(2, "0")}:${String(policy.minute ?? 0).padStart(2, "0")} · ${policy.timezone || "UTC"}`;
}

function createProviderCard(provider, compact = false) {
  const card = node("article", "provider-card");
  card.dataset.providerId = provider.id;

  const header = node("div", "provider-card-header");
  const identity = node("div", "provider-name");
  identity.append(node("span", "provider-glyph", providerGlyph(provider.display_name)));
  const title = node("div");
  title.append(node("h3", "", provider.display_name || provider.id));
  title.append(node("p", "", provider.provider_type || "provider"));
  identity.append(title);
  header.append(identity, statusBadge(provider.status));
  card.append(header);

  const facts = node("dl", "provider-facts");
  const entries = compact ? [
    ["Auth", displayAuth(provider.auth)],
    ["Runtime", `${provider.browser_runtime || "Provider"}${provider.headless === true ? " / Headless" : ""}`],
    ["Keepalive", keepaliveLabel(provider)],
    ["Last success", formatTime(provider.last_success)],
  ] : [
    ["Type", provider.provider_type || "provider"],
    ["Auth", displayAuth(provider.auth)],
    ["Runtime", `${provider.browser_runtime || "Provider"}${provider.headless === true ? " / Headless" : ""}`],
    ["Model alias", provider.id],
    ["Keepalive", keepaliveLabel(provider)],
    ["Last success", formatTime(provider.last_success)],
  ];
  entries.forEach(([label, value]) => {
    facts.append(node("dt", "", label), node("dd", "", value));
  });
  card.append(facts);

  const actions = node("div", "provider-card-actions");
  const detail = node("button", "button button-quiet", "Details");
  detail.type = "button";
  detail.addEventListener("click", () => openProvider(provider.id));
  const test = node("button", "button button-secondary", "Test");
  test.type = "button";
  test.addEventListener("click", () => runProviderAction(provider.id, "test", test));
  const keepalive = node("button", "button button-secondary", "Keepalive");
  keepalive.type = "button";
  keepalive.addEventListener("click", () => runProviderAction(provider.id, "keepalive", keepalive));
  actions.append(detail, test, keepalive);
  card.append(actions);
  return card;
}

function renderProviders() {
  const containers = [byId("dashboard-providers"), byId("providers-list")];
  containers.forEach((container, index) => {
    container.replaceChildren();
    if (!state.providers.length) {
      container.append(node("div", "empty-state", "No providers registered."));
      return;
    }
    state.providers.forEach((provider) => container.append(createProviderCard(provider, index === 0)));
  });

  const modelSelect = byId("playground-model");
  const selected = modelSelect.value;
  modelSelect.replaceChildren();
  state.providers.forEach((provider) => {
    const option = node("option", "", `${provider.display_name || provider.id} · ${provider.id}`);
    option.value = provider.id;
    modelSelect.append(option);
  });
  if ([...modelSelect.options].some((option) => option.value === selected)) modelSelect.value = selected;
}

function renderSummary() {
  const ready = state.providers.filter((provider) => provider.status === "READY").length;
  const authRequired = state.providers.filter((provider) => provider.status === "AUTH_REQUIRED" || provider.auth === "required").length;
  const online = state.providers.length > 0 && ready === state.providers.length;
  byId("summary-service").textContent = online ? "ONLINE" : ready > 0 ? "DEGRADED" : "ATTENTION";
  byId("summary-providers").textContent = String(state.providers.length);
  byId("summary-ready").textContent = String(ready);
  byId("summary-auth").textContent = String(authRequired);
  byId("summary-endpoint").textContent = `${state.service.host}:${state.service.port}`;
  byId("service-label").textContent = online ? "Service OK" : "Service attention";
  const dot = byId("service-pill").querySelector(".status-dot");
  dot.className = `status-dot ${online ? "status-ready" : ready ? "status-degraded" : "status-offline"}`;
  const endpoint = `${state.service.host}:${state.service.port}`;
  byId("sidebar-endpoint").textContent = endpoint;
  byId("footer-endpoint").textContent = endpoint;
  byId("api-base-url").textContent = state.service.base_url;
  byId("last-refreshed").textContent = `Updated ${new Date().toLocaleTimeString()}`;
}

async function refreshProviders({ force = false, quiet = false } = {}) {
  if (state.refreshing) return state.providers;
  state.refreshing = true;
  try {
    const payload = await apiFetch(`/api/providers?refresh=${force ? "true" : "false"}`);
    state.providers = payload.providers || [];
    state.service = payload.service || state.service;
    renderSummary();
    renderProviders();
    return state.providers;
  } catch (error) {
    byId("service-label").textContent = "Backend unreachable";
    const dot = byId("service-pill").querySelector(".status-dot");
    dot.className = "status-dot status-offline";
    if (!quiet) showToast(error.message, true);
    throw error;
  } finally {
    state.refreshing = false;
  }
}

function detailEntry(list, label, value) {
  list.append(node("dt", "", label), node("dd", "", value));
}

async function openProvider(providerId) {
  try {
    const provider = await apiFetch(`/api/providers/${encodeURIComponent(providerId)}?refresh=false`);
    state.activeProvider = provider;
    byId("dialog-title").textContent = provider.display_name || provider.id;
    const status = byId("dialog-status");
    status.replaceChildren(statusBadge(provider.status));
    const details = byId("dialog-details");
    details.replaceChildren();
    [
      ["Provider ID", provider.id],
      ["Display name", provider.display_name],
      ["Status", prettyStatus(provider.status)],
      ["Auth status", displayAuth(provider.auth)],
      ["Provider type", provider.provider_type],
      ["Browser runtime", provider.browser_runtime || "Not applicable"],
      ["Headless", provider.headless === true ? "Yes" : provider.headless === false ? "No" : "Not applicable"],
      ["Profile", provider.profile || "Not applicable"],
      ["Model aliases", (provider.model_aliases || []).join(", ")],
      ["Keepalive enabled", provider.keepalive?.enabled ? "Yes" : "No"],
      ["Keepalive strategy", provider.keepalive?.strategy || "Provider-defined"],
      ["Keepalive schedule", keepaliveLabel(provider)],
      ["Last keepalive", formatTime(provider.last_keepalive)],
      ["Last success", formatTime(provider.last_success)],
      ["Last error", provider.last_error || "None"],
    ].forEach(([label, value]) => detailEntry(details, label, value ?? "—"));

    const actions = byId("dialog-actions");
    actions.replaceChildren();
    const test = node("button", "button button-primary", "Test provider");
    test.type = "button";
    test.addEventListener("click", () => runProviderAction(provider.id, "test", test, true));
    const keepalive = node("button", "button button-secondary", "Run keepalive");
    keepalive.type = "button";
    keepalive.addEventListener("click", () => runProviderAction(provider.id, "keepalive", keepalive, true));
    actions.append(test, keepalive);
    const result = byId("dialog-result");
    result.className = "action-result";
    result.textContent = "";
    byId("provider-dialog").showModal();
  } catch (error) {
    showToast(error.message, true);
  }
}

async function runProviderAction(providerId, action, button, inDialog = false) {
  const key = `${providerId}:${action}`;
  if (state.busyActions.has(key)) return;
  state.busyActions.add(key);
  const original = button.textContent;
  button.disabled = true;
  state.generating = true;
  button.textContent = action === "test" ? "Testing…" : "Running…";
  const resultBox = byId("dialog-result");
  if (inDialog) {
    resultBox.className = "action-result visible";
    resultBox.textContent = action === "test" ? "Sending a real provider test…" : "Running provider keepalive…";
  }
  try {
    const payload = await apiFetch(`/api/providers/${encodeURIComponent(providerId)}/${action}`, { method: "POST" });
    const success = Boolean(payload.success);
    const message = action === "test"
      ? `${success ? "PASS" : "FAIL"}\nResponse: ${payload.response || "—"}\nLatency: ${payload.latency_seconds ?? "—"}s`
      : `${success ? "Keepalive successful" : "Keepalive failed"}\nProvider: ${payload.status || "UNKNOWN"}\nVerified: ${payload.verified ? "Yes" : "No"}`;
    if (inDialog) {
      resultBox.className = `action-result visible ${success ? "success" : "error"}`;
      resultBox.textContent = message;
    }
    showToast(message.replaceAll("\n", " · "), !success);
    await refreshProviders({ force: false, quiet: true });
  } catch (error) {
    if (inDialog) {
      resultBox.className = "action-result visible error";
      resultBox.textContent = error.message;
    }
    showToast(error.message, true);
  } finally {
    state.busyActions.delete(key);
    button.disabled = false;
    button.textContent = original;
  }
}

function showToast(message, isError = false) {
  const toast = node("div", `toast${isError ? " error" : ""}`, message);
  byId("toast-region").append(toast);
  window.setTimeout(() => toast.remove(), 6000);
}

function switchView(viewName) {
  document.querySelectorAll(".view").forEach((view) => view.classList.toggle("active", view.id === `view-${viewName}`));
  document.querySelectorAll(".nav-item").forEach((item) => item.classList.toggle("active", item.dataset.view === viewName));
  history.replaceState(null, "", `#${viewName}`);
}

async function submitPlayground(event) {
  event.preventDefault();
  const prompt = byId("playground-prompt").value.trim();
  const model = byId("playground-model").value;
  if (!prompt || !model) return;
  const button = byId("playground-send");
  const output = byId("playground-response");
  const started = performance.now();
  button.disabled = true;
  button.textContent = "Sending…";
  output.className = "response-output";
  output.textContent = "Waiting for provider response…";
  byId("playground-meta").textContent = `${model} · request in progress`;
  try {
    const payload = await apiFetch("/api/generate", {
      method: "POST",
      body: JSON.stringify({ model, prompt }),
    });
    const elapsed = (performance.now() - started) / 1000;
    output.textContent = payload.text;
    byId("playground-latency").textContent = `${elapsed.toFixed(2)}s`;
    byId("playground-meta").textContent = `${payload.provider} · ${payload.model}`;
    await refreshProviders({ force: false, quiet: true });
  } catch (error) {
    output.className = "response-output error";
    output.textContent = error.message;
    byId("playground-latency").textContent = "failed";
    byId("playground-meta").textContent = "Request failed";
  } finally {
    state.generating = false;
    button.disabled = false;
    button.textContent = "Send request";
  }
}

function bindEvents() {
  document.querySelectorAll(".nav-item").forEach((item) => item.addEventListener("click", () => switchView(item.dataset.view)));
  byId("refresh-dashboard").addEventListener("click", () => refreshProviders({ force: true }));
  byId("refresh-providers").addEventListener("click", () => refreshProviders({ force: true }));
  byId("dialog-close").addEventListener("click", () => byId("provider-dialog").close());
  byId("provider-dialog").addEventListener("click", (event) => {
    if (event.target === byId("provider-dialog")) byId("provider-dialog").close();
  });
  byId("playground-form").addEventListener("submit", submitPlayground);
  byId("playground-prompt").addEventListener("input", (event) => {
    byId("prompt-count").textContent = `${event.target.value.length.toLocaleString()} characters`;
  });
  document.querySelectorAll("[data-copy-target]").forEach((button) => button.addEventListener("click", async () => {
    const target = byId(button.dataset.copyTarget);
    try {
      await navigator.clipboard.writeText(target.textContent);
      const original = button.textContent;
      button.textContent = "Copied";
      window.setTimeout(() => { button.textContent = original; }, 1200);
    } catch {
      showToast("Clipboard access is unavailable", true);
    }
  }));
}

document.addEventListener("DOMContentLoaded", async () => {
  bindEvents();
  const initialView = location.hash.slice(1);
  if (["dashboard", "providers", "api", "playground"].includes(initialView)) switchView(initialView);
  try { await refreshProviders({ force: true }); } catch { /* Visible service state already updated. */ }
  window.setInterval(() => {
    if (!document.hidden && !state.generating && state.busyActions.size === 0) {
      refreshProviders({ force: true, quiet: true }).catch(() => {});
    }
  }, 30000);
});
