const select = document.getElementById("category-select");
const list = document.getElementById("digest-list");
const dateEl = document.getElementById("digest-date");
const searchForm = document.getElementById("search-form");
const searchInput = document.getElementById("search-input");
const STORAGE_KEY = "sci-news-last-category";

function slugify(text) {
  return text
    .toLowerCase()
    .trim()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

searchForm.addEventListener("submit", (e) => {
  e.preventDefault();
  const query = searchInput.value.trim();
  if (!query) return;
  const slug = slugify(query);
  window.location.href = `/digest.html?q=${encodeURIComponent(slug)}`;
});

function formatDate(iso) {
  if (!iso) return "";
  const d = new Date(iso + "T00:00:00");
  return d.toLocaleDateString("en-US", { year: "numeric", month: "long", day: "numeric" });
}

function renderEntries(digest) {
  list.innerHTML = "";

  if (!digest || !digest.entries || digest.entries.length === 0) {
    list.innerHTML = `<p class="state-message">No digest for this field yet. Run the pipeline (<code>python main.py</code>) to generate one.</p>`;
    dateEl.textContent = "";
    return;
  }

  dateEl.textContent = `Week of ${formatDate(digest.date)}`;

  digest.entries.forEach((entry) => {
    const article = document.createElement("article");
    article.className = "entry";
    article.innerHTML = `
      <h2 class="entry-title"><a href="${entry.link}" target="_blank" rel="noopener">${entry.title}</a></h2>
      <p class="entry-paragraph">${entry.paragraph}</p>
      <p class="entry-link"><a href="${entry.link}" target="_blank" rel="noopener">Read the paper</a></p>
    `;
    list.appendChild(article);
  });
}

async function loadCategory(category) {
  list.innerHTML = `<p class="state-message">Loading…</p>`;
  try {
    const res = await fetch(`digests/${category}.json?_=${Date.now()}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const digest = await res.json();
    renderEntries(digest);
  } catch (err) {
    list.innerHTML = `<p class="state-message">Couldn't load the ${category} digest yet — the pipeline may not have run for this field.</p>`;
    dateEl.textContent = "";
  }
}

select.addEventListener("change", () => {
  const category = select.value;
  localStorage.setItem(STORAGE_KEY, category);
  loadCategory(category);
});

// Restore the visitor's last-viewed field, if any, then load it.
const lastCategory = localStorage.getItem(STORAGE_KEY);
if (lastCategory && [...select.options].some((o) => o.value === lastCategory)) {
  select.value = lastCategory;
}
loadCategory(select.value);
