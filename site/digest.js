const container = document.getElementById("digest-container");
const API_BASE_URL = (window.SCI_NEWS_CONFIG?.apiBaseUrl || "http://localhost:8000").replace(/\/+$/, "");

function escapeHTML(value) {
  return String(value ?? "").replace(/[&<>"']/g, (character) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;"
  })[character]);
}

function safeExternalUrl(value) {
  try {
    const url = new URL(String(value || ""), window.location.href);
    if (url.protocol === "http:" || url.protocol === "https:") return url.href;
  } catch (_) {
    // Invalid URLs are rendered as plain text instead of links.
  }
  return "";
}

function formatDate(iso) {
  if (!iso) return "";
  const date = new Date(`${iso}T00:00:00`);
  return Number.isNaN(date.getTime())
    ? ""
    : date.toLocaleDateString("en-US", { year: "numeric", month: "long", day: "numeric" });
}

function formatAuthors(authors) {
  if (!Array.isArray(authors) || authors.length === 0) return "";
  const names = authors.slice(0, 3).join(", ");
  return escapeHTML(authors.length <= 3 ? names : `${names} et al.`);
}

function renderEntries(entries, query, metaExtra = "") {
  if (!Array.isArray(entries) || entries.length === 0) {
    container.innerHTML = `
      <div class="state-message">
        <p>No matching research summaries were produced for “${escapeHTML(query)}”.</p>
        <p>Try a more specific or broader topic, or choose one of the example searches on the homepage.</p>
      </div>
    `;
    return;
  }

  const entriesHTML = entries.map((entry) => {
    const title = escapeHTML(entry.title || "Untitled research");
    const paragraph = escapeHTML(entry.paragraph || "");
    const sourceText = escapeHTML(entry.source || "Research");
    const sourceClass = String(entry.source || "research").toLowerCase().replace(/[^a-z0-9_-]/g, "");
    const published = escapeHTML(formatDate(entry.published));
    const href = safeExternalUrl(entry.link);
    const titleMarkup = href
      ? `<a href="${escapeHTML(href)}" target="_blank" rel="noopener noreferrer">${title}</a>`
      : title;
    const linkMarkup = href
      ? `<a href="${escapeHTML(href)}" target="_blank" rel="noopener noreferrer" class="entry-link">Read the paper →</a>`
      : `<span class="entry-link">Source link unavailable</span>`;

    return `
      <article class="entry-card">
        <div class="entry-header">
          <h2 class="entry-title">${titleMarkup}</h2>
          <span class="entry-source ${sourceClass}">${sourceText}</span>
        </div>
        <p class="entry-authors">${formatAuthors(entry.authors)}</p>
        <p class="entry-paragraph">${paragraph}</p>
        <div class="entry-footer">
          <span>${published}</span>
          ${linkMarkup}
        </div>
      </article>
    `;
  }).join("");

  container.innerHTML = `
    <div class="digest-header">
      <h1>${escapeHTML(query)}</h1>
      <div class="digest-meta">
        <span>${escapeHTML(metaExtra)}</span>
        <span>${entries.length} research summaries</span>
      </div>
    </div>
    <div class="entry-list">
      ${entriesHTML}
    </div>
  `;
}

function showError(query, message) {
  container.innerHTML = `
    <div class="state-message">
      <p>Could not generate a digest for “${escapeHTML(query)}”.</p>
      <p>${escapeHTML(message || "Please try again later.")}</p>
      <p><a href="./">← Back to search</a></p>
    </div>
  `;
}

function showLoading(message) {
  container.innerHTML = `
    <div class="digest-loading">
      <div class="spinner"></div>
      <p class="loading-message">${escapeHTML(message)}</p>
    </div>
  `;
}

async function loadStaticDigest(slug) {
  const response = await fetch(`digests/search/${encodeURIComponent(slug)}.json?_=${Date.now()}`);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
}

async function startResearchJob(query) {
  const response = await fetch(`${API_BASE_URL}/research`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query })
  });

  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    throw new Error(data.detail || `API error ${response.status}`);
  }
  return response.json();
}

async function pollJob(jobId) {
  const response = await fetch(`${API_BASE_URL}/research/${encodeURIComponent(jobId)}`);
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    throw new Error(data.detail || `API error ${response.status}`);
  }
  return response.json();
}

async function loadDigest() {
  const params = new URLSearchParams(window.location.search);
  const query = (params.get("q") || "").trim();
  const slug = query.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 80);

  if (!slug) {
    window.location.href = "./";
    return;
  }

  const loadingMessages = [
    "Reading the literature…",
    "Finding relevant research…",
    "Preparing your digest…"
  ];
  let messageIndex = 0;
  showLoading(loadingMessages[messageIndex]);
  const messageTimer = setInterval(() => {
    messageIndex = (messageIndex + 1) % loadingMessages.length;
    const message = document.querySelector(".loading-message");
    if (message) message.textContent = loadingMessages[messageIndex];
  }, 1800);

  try {
    // Pre-generated examples load immediately without requiring the API.
    try {
      const digest = await loadStaticDigest(slug);
      renderEntries(digest.entries, query, formatDate(digest.date));
      return;
    } catch (_) {
      // No static copy exists; continue to live research.
    }

    const created = await startResearchJob(query);
    let jobData;
    for (let attempt = 0; attempt < 100; attempt += 1) {
      jobData = await pollJob(created.job_id);

      if (jobData.status === "failed") {
        throw new Error(jobData.error || "The research job failed.");
      }

      if (jobData.status === "completed") {
        const result = jobData.result || {};
        if (result.status && result.status !== "completed") {
          throw new Error(result.error || `The search finished with status: ${result.status}.`);
        }
        const entries = result.digest_entries || [];
        const verification = result.verification_status
          ? `Fact-check: ${result.verification_status}`
          : "";
        renderEntries(entries, query, verification);
        return;
      }

      await new Promise((resolve) => setTimeout(resolve, 3000));
    }

    throw new Error("The research job is taking too long. Please try again later.");
  } catch (error) {
    showError(query, error?.message || "An unexpected error occurred.");
  } finally {
    clearInterval(messageTimer);
  }
}

loadDigest();
