/* InsurFlow AI Dashboard — vanilla JS, no frameworks */

const statusEl = document.getElementById("status");
const rowsEl = document.getElementById("rows");
const refreshBtn = document.getElementById("refreshBtn");
const loadingEl = document.getElementById("loading");
const toastEl = document.getElementById("toast");

function setLoading(on) {
  loadingEl.classList.toggle("show", Boolean(on));
  refreshBtn.disabled = Boolean(on);
}

function showToast(message, isError = false) {
  toastEl.textContent = message;
  toastEl.classList.add("show");
  toastEl.classList.toggle("error", Boolean(isError));
  clearTimeout(showToast._t);
  showToast._t = setTimeout(() => toastEl.classList.remove("show"), 4500);
}

function setStatus(text, isError = false) {
  statusEl.textContent = text;
  statusEl.style.color = isError ? "#991b1b" : "var(--muted)";
}

function fmt3(x) {
  if (x === null || x === undefined || Number.isNaN(Number(x))) return "—";
  return Number(x).toFixed(3);
}

function esc(s) {
  return String(s ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function fraudClass(score) {
  const s = Number(score);
  if (Number.isNaN(s)) return "score-mid";
  if (s >= 0.66) return "score-high";
  if (s >= 0.33) return "score-mid";
  return "score-low";
}

function decisionClass(d) {
  const u = String(d || "").toUpperCase();
  if (u === "APPROVED") return "approved";
  if (u === "REJECTED") return "rejected";
  if (u === "INVESTIGATE") return "investigate";
  return "";
}

function renderExplanation(text) {
  const t = String(text || "").trim();
  if (!t) return '<div class="explanation muted">No explanation stored.</div>';
  try {
    const j = JSON.parse(t);
    if (j && typeof j === "object" && (j.summary || j.key_factors)) {
      const sum = j.summary ? `<div class="expl-summary">${esc(j.summary)}</div>` : "";
      const factors = Array.isArray(j.key_factors)
        ? `<ul class="expl-factors">${j.key_factors.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>`
        : "";
      const ref = j.similar_case_reference
        ? `<div class="muted expl-ref">${esc(j.similar_case_reference)}</div>`
        : "";
      return `<div class="explanation expl-structured">${sum}${factors}${ref}</div>`;
    }
  } catch {
    /* fall through */
  }
  return `<div class="explanation">${esc(t)}</div>`;
}

function renderEntities(obj) {
  if (!obj || typeof obj !== "object") return "";
  try {
    const s = JSON.stringify(obj).slice(0, 400);
    return `<div class="entities">${esc(s)}</div>`;
  } catch {
    return "";
  }
}

async function apiJson(path, options) {
  const res = await fetch(path, options);
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`HTTP ${res.status} ${res.statusText}${text ? ` — ${text}` : ""}`);
  }
  return await res.json();
}

async function review(claimId, action) {
  setLoading(true);
  setStatus("");
  try {
    await apiJson(`/claims/${encodeURIComponent(claimId)}/review`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action }),
    });
    showToast(`Review saved: ${action}`, false);
    await refresh(false);
  } catch (err) {
    setStatus(String(err?.message || err), true);
    showToast(String(err?.message || err), true);
  } finally {
    setLoading(false);
  }
}

function render(items) {
  rowsEl.innerHTML = "";
  for (const it of items) {
    const hitlNeeded = Boolean(it.hitl_needed);
    const reviewed = it.review_status || it.reviewed_action;
    const reviewStatusHtml = reviewed
      ? `<div class="review-status done"><strong>${esc(reviewed)}</strong>`
        + (it.reviewed_at ? `<div class="muted">${esc(it.reviewed_at)}</div>` : "")
        + (it.reviewed_by ? `<div class="muted">by ${esc(it.reviewed_by)}</div>` : "")
        + `</div>`
      : `<span class="muted">—</span>`;

    const hitlPill = hitlNeeded
      ? `<span class="pill hitl">Needs review</span>`
      : `<span class="pill ok">OK</span>`;

    const canAct = hitlNeeded && !reviewed;
    const actionsHtml = canAct
      ? `<div class="cell-actions">
           <button type="button" data-action="APPROVED" data-claim="${esc(it.claim_id)}">Approve</button>
           <button type="button" data-action="REJECTED" data-claim="${esc(it.claim_id)}">Reject</button>
         </div>`
      : `<span class="muted">—</span>`;

    const preview = (it.claim_description || "").slice(0, 160);
    const expl = it.explanation || "";
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>
        <div class="claim-id">${esc(it.claim_id)}</div>
        <div class="preview">${esc(preview)}${(it.claim_description || "").length > 160 ? "…" : ""}</div>
        <details class="explain-details">
          <summary>Why flagged / full explanation</summary>
          ${renderExplanation(expl)}
          ${renderEntities(it.entities)}
        </details>
      </td>
      <td>
        <span class="score-pill ${fraudClass(it.fraud_score)}">${fmt3(it.fraud_score)}</span>
        <div class="muted" style="margin-top:6px">${hitlPill}</div>
      </td>
      <td>
        <span class="decision ${decisionClass(it.decision)}">${esc(it.decision || "—")}</span>
      </td>
      <td>${fmt3(it.confidence)}</td>
      <td>${reviewStatusHtml}</td>
      <td>${actionsHtml}</td>
    `;
    rowsEl.appendChild(tr);
  }

  rowsEl.querySelectorAll("button[data-action][data-claim]").forEach((btn) => {
    btn.addEventListener("click", async (e) => {
      const action = e.currentTarget.getAttribute("data-action");
      const claimId = e.currentTarget.getAttribute("data-claim");
      if (!action || !claimId) return;
      await review(claimId, action);
    });
  });
}

async function refresh(showLoading = true) {
  if (showLoading) setLoading(true);
  setStatus("");
  try {
    const items = await apiJson("/claims");
    render(items);
    setStatus(`Loaded ${items.length} claim(s).`);
  } catch (err) {
    setStatus(String(err?.message || err), true);
    showToast(String(err?.message || err), true);
  } finally {
    if (showLoading) setLoading(false);
  }
}

refreshBtn.addEventListener("click", () => refresh(true));
refresh(true);
