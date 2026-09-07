const container = document.getElementById("digest-container");

// ---------------------------------------------------------------------------
// API configuration (adjust for production)
// ---------------------------------------------------------------------------
const API_BASE_URL = "http://localhost:8000"; // change to your deployed backend
const API_KEY = ""; // if backend requires X-API-Key, set it here

function formatDate(iso) {
  if (!iso) return "";
  const d = new Date(iso + "T00:00:00");
  return d.toLocaleDateString("en-US", { year: "numeric", month: "long", day: "numeric" });
}

function formatAuthors(authors) {
  if (!authors || authors.length === 0) return "";
  if (authors.length <= 3) return authors.join(", ");
  return `${authors.slice(0, 3).join(", ")} et al.`;
}

function renderEntries(entries, query, metaExtra = "") {
  if (!entries || entries.length === 0) {
    container.innerHTML = `
      <div class="state-message">
        <p>No digest found for “${query}”.</p>
        <p>Try one of the examples from the homepage, or run the pipeline locally to generate this digest.</p>
      </div>
    `;
    return;
  }

  const entriesHTML = entries.map((entry) => `
    <article class="entry-card">
      <div class="entry-header">
        <h2 class="entry-title"><a href="${entry.link}" target="_blank" rel="noopener">${entry.title}</a></h2>
        <span class="entry-source ${entry.source}">${entry.source}</span>
      </div>
      <p class="entry-authors">${formatAuthors(entry.authors)}</p>
      <p class="entry-paragraph">${entry.paragraph}</p>
      <div class="entry-footer">
        <span>${formatDate(entry.published)}</span>
        <a href="${entry.link}" target="_blank" rel="noopener" class="entry-link">Read the paper →</a>
      </div>
    </article>
  `).join("");

  container.innerHTML = `
    <div class="digest-header">
      <h1>${query}</h1>
      <div class="digest-meta">
        <span>${metaExtra || ""}</span>
        <span>${entries.length} selected papers</span>
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
      <p>Could not load the digest for “${query}”.</p>
      <p>${message || "The file might not exist yet. Please try again later or run the pipeline."}</p>
      <p><a href="/">← Back to search</a></p>
    </div>
  `;
}

function showLoading(message) {
  container.innerHTML = `
    <div class="digest-loading">
      <div class="spinner"></div>
      <p class="loading-message">${message}</p>
    </div>
  `;
}

// ---------------------------------------------------------------------------
// Static file loading (existing pre-generated digests)
// ---------------------------------------------------------------------------
async function loadStaticDigest(slug, query) {
  const res = await fetch(`digests/search/${slug}.json?_=${Date.now()}`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return await res.json();
}

// ---------------------------------------------------------------------------
// API interaction (new dynamic searches)
// ---------------------------------------------------------------------------
async function startResearchJob(query) {
  const headers = { "Content-Type": "application/json" };
  if (API_KEY) headers["X-API-Key"] = API_KEY;

  const response = await fetch(`${API_BASE_URL}/research`, {
    method: "POST",
    headers,
    body: JSON.stringify({ query }),
  });

  if (!response.ok) {
    const errData = await response.json().catch(() => ({}));
    throw new Error(errData.detail || `API error ${response.status}`);
  }
  const data = await response.json();
  return data.job_id;
}

async function pollJob(jobId) {
  const headers = {};
  if (API_KEY) headers["X-API-Key"] = API_KEY;

  const response = await fetch(`${API_BASE_URL}/research/${jobId}`, { headers });
  if (!response.ok) {
    const errData = await response.json().catch(() => ({}));
    throw new Error(errData.detail || `API error ${response.status}`);
  }
  return await response.json();
}

async function loadDigest() {
  const params = new URLSearchParams(window.location.search);
  const query = params.get("q") || "";
  const slug = query.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "");

  if (!slug) {
    window.location.href = "/";
    return;
  }

  const loadingMessages = [
    "Reading the literature…",
    "Finding relevant research…",
    "Preparing your digest…",
  ];
  let msgIndex = 0;
  showLoading(loadingMessages[msgIndex]);
  const interval = setInterval(() => {
    msgIndex = (msgIndex + 1) % loadingMessages.length;
    const msgEl = document.querySelector(".loading-message");
    if (msgEl) msgEl.textContent = loadingMessages[msgIndex];
  }, 1500);

  try {
    // 1. Try to load a pre-generated static digest (fast path)
    try {
      const staticDigest = await loadStaticDigest(slug, query);
      clearInterval(interval);
      renderEntries(staticDigest.entries, query, formatDate(staticDigest.date));
      return;
    } catch (staticErr) {
      // Static file not found; continue to API
    }

    // 2. Static file missing → start a new research job via API
    const jobId = await startResearchJob(query);
    let jobData;
    const maxAttempts = 100; // approx 5 minutes with 3s polling
    for (let attempt = 0; attempt < maxAttempts; attempt++) {
      jobData = await pollJob(jobId);
      if (jobData.status === "completed") {
        clearInterval(interval);
        // The API result contains `result.digest_entries` and `result.verification_status`
        const entries = jobData.result?.digest_entries || [];
        const meta = jobData.result?.verification_status
          ? `Verification: ${jobData.result.verification_status}`
          : "";
        renderEntries(entries, query, meta);
        return;
      } else if (jobData.status === "failed") {
        clearInterval(interval);
        showError(query, jobData.error || "Research job failed.");
        return;
      }
      // Still queued or running; wait and poll again
      await new Promise((resolve) => setTimeout(resolve, 3000));
    }

    // Timeout
    clearInterval(interval);
    showError(query, "The research job is taking too long. Please try again later.");
  } catch (err) {
    clearInterval(interval);
    showError(query, err.message || "An unexpected error occurred.");
  }
}

loadDigest();