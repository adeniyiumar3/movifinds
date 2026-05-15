// Results page logic: loads last search state, fetches recommendations, and renders them.

function getSavedSearch() {
  const json = localStorage.getItem("moviFindsLastSearch");
  if (!json) return null;
  try {
    return JSON.parse(json);
  } catch (err) {
    return null;
  }
}

function renderMovies(movies) {
  const list = document.getElementById("results-list");
  list.innerHTML = "";

  if (!movies.length) {
    list.innerHTML = `<div class="error-msg">No movies matched your filters. Try relaxing your language/year selection.</div>`;
    return;
  }

  const palettes = [
    "#1a0c28","#0c1a10","#1a100a","#0a0c1a","#1a0a0a",
    "#0a1418","#180a1a","#0a180a","#1a1a0a","#0a0a18"
  ];

  movies.forEach((m, i) => {
    const tags  = (m.tags || []).slice(0, 4).map(t => `<span class="tag">${t}</span>`).join("");
    const simPct = Math.round(m.similarity || 0);
    const color  = palettes[i % palettes.length];
    const row    = document.createElement("div");
    row.className = "movie-row";
    row.innerHTML = `
      <div class="movie-rank">${m.rank}</div>
      <div class="movie-thumb" style="background:${color}"></div>
      <div class="movie-info">
        <div class="movie-title-text">${m.title} <span class="movie-year">${m.year || ""}</span></div>
        <div class="movie-desc">${m.description || "No description available."}</div>
        <div class="movie-tags">${tags}</div>
        <span class="confidence-badge">${m.confidence || "Match"} · ${simPct}% similarity</span>
      </div>
      <div class="movie-score">
        <div class="score-num">${m.score || "—"}</div>
        <span class="score-max">/ 10</span>
        <div class="similarity-bar">
          <div class="similarity-fill" style="width:${simPct}%"></div>
        </div>
      </div>
    `;
    list.appendChild(row);
  });
}

function showError(msg) {
  const error = document.getElementById("results-error");
  error.style.display = "block";
  error.textContent = `⚠️ ${msg}`;
}

async function loadResults() {
  const saved = getSavedSearch();
  const loading = document.getElementById("results-loading");
  const queryPreview = document.getElementById("query-preview");

  if (!saved) {
    loading.style.display = "none";
    queryPreview.textContent = "No search state found. Please refine your search first.";
    showError("Start from the selection page so your filters are applied.");
    return;
  }

  const queryText = saved.query || "Your last search";
  queryPreview.textContent = queryText;

  try {
    const res = await fetch("/recommend", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(saved.payload)
    });
    const data = await res.json();

    loading.style.display = "none";
    if (!data.success) {
      return showError(data.error || "Failed to load results.");
    }

    renderMovies(data.movies);
  } catch (err) {
    loading.style.display = "none";
    showError("Cannot connect to server. Is app.py running?");
  }
}

function signOut() {
  localStorage.removeItem("moviFindsLastSearch");
  window.location.href = "/signin";
}

document.addEventListener("DOMContentLoaded", loadResults);
