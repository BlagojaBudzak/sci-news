const searchForm = document.getElementById("search-form");
const searchInput = document.getElementById("search-input");

searchForm.addEventListener("submit", (e) => {
  e.preventDefault();
  const query = searchInput.value.trim();
  if (!query) return;
  // Pass the raw query; digest.js will slugify for static fallback.
  window.location.href = `digest.html?q=${encodeURIComponent(query)}`;
});