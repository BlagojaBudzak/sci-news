const container = document.getElementById("digest-container");

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

function renderDigest(digest, query) {
  if (!digest || !digest.entries || digest.entries.length === 0) {
    container.innerHTML = `
      <div class="state-message">
        <p>No digest found for “${query}”.</p>
        <p>Try one of the examples from the homepage, or run the pipeline locally to generate this digest.</p>
      </div>
    `;
    return;
  }

  const entriesHTML = digest.entries.map((entry) => `
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
        <span>${formatDate(digest.date)}</span>
        <span>${digest.entries.length} selected papers</span>
      </div>
    </div>
    <div class="entry-list">
      ${entriesHTML}
    </div>
  `;
}

function showError(query) {
  container.innerHTML = `
    <div class="state-message">
      <p>Could not load the digest for “${query}”.</p>
      <p>The file might not exist yet. Please try again later or run the pipeline.</p>
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
    const res = await fetch(`digests/search/${slug}.json?_=${Date.now()}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const digest = await res.json();
    clearInterval(interval);
    renderDigest(digest, query);
  } catch (err) {
    clearInterval(interval);
    showError(query);
  }
}

loadDigest();