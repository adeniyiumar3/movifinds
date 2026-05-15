// ============================================================
// main.js — MoviFinds LOCAL Frontend
//
// What this does:
// 1. On page load: fetches genres/languages/years from
//    the Flask backend (which reads them from your CSV)
// 2. Dynamically builds the filter chips and dropdowns
// 3. Tracks selections (max 5 genres, 2 for other categories)
// 4. Builds live search query preview
// 5. Sends request to Flask → recommender.py → returns results
// 6. Renders results with confidence scores + similarity bars
// ============================================================

// ── App State ────────────────────────────────────────────────
const state = {
  genre:       [],      // up to 5
  mood:        [],      // up to 2
  situation:   [],      // up to 2
  language:    "",
  year:        "",
  format:      "movie",
  audience:    "Teen & Young Adult",
  similarTo:   "",
  customText:  ""
};


// ── ON PAGE LOAD: fetch filters from backend ─────────────────
document.addEventListener("DOMContentLoaded", async () => {
  try {
    const res  = await fetch("/filters");
    const data = await res.json();

    if (!data.success) throw new Error(data.error);

    // Populate genre chips dynamically
    buildGenreChips(data.genres);

    // Populate language dropdown
    buildSelect("language-select", data.languages);

    // Populate year-range dropdown
    buildSelect("year-select", data.yearRanges);

    // Update dataset stats card
    document.getElementById("stat-genres").textContent    = data.genres.length;
    document.getElementById("stat-languages").textContent = data.languages.length;
    document.getElementById("stat-countries").textContent = 88;
    // Movie count comes from total (set by backend if available)
    if (data.total_movies) {
      document.getElementById("stat-movies").textContent = data.total_movies.toLocaleString();
    }

    // Restore the last search state if available
    loadSavedState();
    updateCTAState();

    // Hide loading, show the grid
    document.getElementById("filters-loading").style.display = "none";
    document.getElementById("main-grid").style.display = "grid";

  } catch (err) {
    document.getElementById("filters-loading").innerHTML =
      `<p style="color:#e05c2a">❌ Could not load filters: ${err.message}<br>Make sure app.py is running and your CSV is in the data/ folder.</p>`;
  }
});


// ── BUILD GENRE CHIPS FROM DATASET ──────────────────────────
function buildGenreChips(genres) {
  const container = document.getElementById("genre-chips");
  container.innerHTML = "";
  genres.forEach(genre => {
    const chip = document.createElement("div");
    chip.className   = "chip";
    chip.dataset.val = genre;
    chip.textContent = genre;
    container.appendChild(chip);
  });
}


// ── BUILD SELECT DROPDOWN ────────────────────────────────────
function buildSelect(elementId, options) {
  const select = document.getElementById(elementId);
  // Keep the first "Any ..." option
  const first  = select.options[0];
  select.innerHTML = "";
  select.appendChild(first);
  options.forEach(opt => {
    const el   = document.createElement("option");
    el.value   = opt;
    el.textContent = opt;
    select.appendChild(el);
  });

  if (elementId === "year-select" && options.length > 0) {
    const firstYearOption = select.options[1];
    if (firstYearOption && firstYearOption.value.includes("-")) {
      const startYear = firstYearOption.value.split("-")[0];
      firstYearOption.textContent = `${startYear} till date`;
    }
  }
}

function saveSearchState() {
  const payload = {
    genre:        state.genre,
    mood:         state.mood,
    situation:    state.situation,
    language:     state.language,
    year:         state.year,
    format:       state.format,
    audience:     state.audience,
    similar_to:   state.similarTo,
    custom_text:  state.customText
  };

  const query = buildQueryText();
  localStorage.setItem("moviFindsLastSearch", JSON.stringify({
    payload,
    query: query || "Movie search"
  }));
}

function signOut() {
  localStorage.removeItem("moviFindsLastSearch");
  window.location.href = "/signin";
}

function loadSavedState() {
  const raw = localStorage.getItem("moviFindsLastSearch");
  if (!raw) return;

  let saved;
  try { saved = JSON.parse(raw); } catch (err) { return; }
  if (!saved || !saved.payload) return;

  const payload = saved.payload;
  state.genre      = payload.genre || [];
  state.mood       = payload.mood || [];
  state.situation  = payload.situation || [];
  state.language   = payload.language || "";
  state.year       = payload.year || "";
  state.format     = payload.format || "movie";
  state.audience   = payload.audience || "Teen & Young Adult";
  state.similarTo  = payload.similar_to || "";
  state.customText = payload.custom_text || "";

  document.querySelectorAll(".chips[data-group]").forEach(container => {
    const group    = container.dataset.group;
    const selected = state[group] || [];
    const max      = parseInt(container.dataset.max) || 2;

    container.querySelectorAll(".chip").forEach(chip => {
      const val = chip.dataset.val;
      if (selected.includes(val)) {
        chip.classList.add("sel");
      } else {
        chip.classList.remove("sel");
      }
    });

    container.querySelectorAll(".chip").forEach(chip => {
      if (!chip.classList.contains("sel")) {
        chip.classList.toggle("disabled", selected.length >= max);
      }
    });

    const counter = document.getElementById("counter-" + group);
    if (counter) {
      counter.textContent = `${selected.length}/${max}`;
      counter.classList.toggle("full", selected.length >= max);
    }
  });

  const languageSelect = document.getElementById("language-select");
  if (languageSelect) languageSelect.value = state.language;
  const yearSelect = document.getElementById("year-select");
  if (yearSelect) yearSelect.value = state.year;

  document.querySelectorAll(".seg-btn").forEach(btn => {
    btn.classList.toggle("sel", btn.dataset.val === state.format);
  });

  document.querySelectorAll(".radio-item").forEach(item => {
    const label = item.querySelector(".radio-label")?.textContent.trim();
    item.classList.toggle("sel", label === state.audience);
  });

  const similarInput = document.getElementById("similar-input");
  if (similarInput) similarInput.value = state.similarTo;
  const customInput = document.getElementById("custom-input");
  if (customInput) customInput.value = state.customText;

  updatePrompt();
}


// ── CHIP CLICK HANDLER ───────────────────────────────────────
// Works for all chip groups. Reads data-max attribute for limit.
document.addEventListener("click", e => {
  const chip = e.target.closest(".chip");
  if (!chip || chip.classList.contains("disabled")) return;

  const container = chip.closest(".chips[data-group]");
  if (!container) return;

  const group = container.dataset.group;
  const max   = parseInt(container.dataset.max) || 2;
  const val   = chip.dataset.val;
  const sel   = state[group];

  if (chip.classList.contains("sel")) {
    // Deselect
    sel.splice(sel.indexOf(val), 1);
    chip.classList.remove("sel");
  } else {
    if (sel.length >= max) return;
    sel.push(val);
    chip.classList.add("sel");
  }

  // Update disabled state
  container.querySelectorAll(".chip").forEach(c => {
    if (!c.classList.contains("sel")) {
      c.classList.toggle("disabled", sel.length >= max);
    }
  });

  // Update counter
  const counter = document.getElementById("counter-" + group);
  if (counter) {
    counter.textContent = `${sel.length}/${max}`;
    counter.classList.toggle("full", sel.length >= max);
  }

  updatePrompt();
});


// ── FORMAT SELECTOR ──────────────────────────────────────────
function segClick(btn) {
  btn.closest(".segment").querySelectorAll(".seg-btn")
     .forEach(b => b.classList.remove("sel"));
  btn.classList.add("sel");
  state.format = btn.dataset.val;
  updatePrompt();
}


// ── AUDIENCE RADIO ───────────────────────────────────────────
function pickRadio(el, val) {
  el.closest(".radio-list").querySelectorAll(".radio-item")
    .forEach(r => r.classList.remove("sel"));
  el.classList.add("sel");
  state.audience = val;
  updatePrompt();
}


// ── LANGUAGE / YEAR dropdowns ─────────────────────────────
document.getElementById("language-select").addEventListener("change", function() {
  state.language = this.value;
  updatePrompt();
});
document.getElementById("year-select").addEventListener("change", function() {
  state.year = this.value;
  updatePrompt();
});


// ── SIMILAR TO + CUSTOM TEXT ─────────────────────────────────
document.getElementById("similar-input").addEventListener("input", function() {
  state.similarTo = this.value;
  updatePrompt();
});
document.getElementById("custom-input").addEventListener("input", function() {
  state.customText = this.value;
  updatePrompt();
});


// ── LIVE QUERY PREVIEW ───────────────────────────────────────
function buildQueryText() {
  const { genre, mood, situation,
          language, year, format, audience,
          similarTo, customText } = state;

  const hasSelection = genre.length || mood.length || situation.length ||
                       language || year || similarTo || customText ||
                       format !== "movie" || audience !== "Teen & Young Adult";
  if (!hasSelection) return null;

  const join = arr => arr.join(" and ");
  const parts = [];

  if (genre.length)     parts.push(`Genre: ${join(genre)}.`);
  if (mood.length)      parts.push(`Mood: ${join(mood)}.`);
  if (situation.length) parts.push(`For: ${join(situation)}.`);
  if (language)         parts.push(`Language: ${language}.`);
  if (year)             parts.push(`Release year: ${year}.`);
  if (format)           parts.push(`Format: ${format}.`);
  if (audience)         parts.push(`Audience: ${audience}.`);

  if (similarTo)    parts.push(`Similar to: "${similarTo}".`);
  if (customText.trim()) parts.push(customText.trim());

  return parts.join(" ");
}

function hasSelectedCriteria() {
  const { genre, mood, situation, language, year,
          similarTo, customText, format, audience } = state;
  return genre.length || mood.length || situation.length ||
         language || year || similarTo.trim() || customText.trim() ||
         format !== "movie" || audience !== "Teen & Young Adult";
}

function updateCTAState() {
  const btn = document.getElementById("cta-btn");
  if (!btn) return;
  btn.disabled = !hasSelectedCriteria();
}

function updatePrompt() {
  const text = buildQueryText();
  const el   = document.getElementById("prompt-preview");
  if (text) {
    el.textContent = `"${text}"`;
    el.classList.remove("empty");
  } else {
    el.textContent = "Select preferences above to see your search query...";
    el.classList.add("empty");
  }
  updateCTAState();
}


// ── CLEAR ALL ────────────────────────────────────────────────
function clearAll() {
  document.querySelectorAll(".chip").forEach(c => c.classList.remove("sel", "disabled"));
  document.querySelectorAll(".counter").forEach(c => {
    const max = c.id.includes("genre") ? "5" : "2";
    c.textContent = `0/${max}`;
    c.classList.remove("full");
  });
  ["genre","mood","situation"].forEach(k => state[k] = []);

  document.getElementById("custom-input").value    = "";
  document.getElementById("similar-input").value   = "";
  document.getElementById("language-select").value = "";
  document.getElementById("year-select").value  = "";

  state.customText  = "";
  state.similarTo   = "";
  state.language    = "";
  state.year       = "";
  localStorage.removeItem("moviFindsLastSearch");
  updatePrompt();
  updateCTAState();
}


// ── GET RECOMMENDATIONS ──────────────────────────────────────
function getRecommendations() {
  if (!hasSelectedCriteria()) {
    const el = document.getElementById("prompt-preview");
    if (el) {
      el.textContent = "Please choose at least one preference before getting recommendations.";
      el.classList.remove("empty");
    }
    updateCTAState();
    return;
  }

  saveSearchState();
  window.location.href = "/results";
}


// ── RENDER MOVIE RESULTS ─────────────────────────────────────
function renderMovies(movies) {
  const list = document.getElementById("results-list");
  list.innerHTML = "";

  if (!movies.length) {
    list.innerHTML = `<div class="error-msg">No movies matched your filters. Try relaxing your language/year selection.</div>`;
    list.style.display = "block";
    return;
  }

  // Color palette for poster placeholders
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

  list.style.display = "block";
}


// ── ERROR STATE ──────────────────────────────────────────────
function showError(msg) {
  const list = document.getElementById("results-list");
  list.innerHTML = `<div class="error-msg">⚠️ ${msg}</div>`;
  list.style.display = "block";
}


// ── LOGOUT HANDLER ───────────────────────────────────────────
function signOut() {
  localStorage.removeItem("moviFindsLastSearch");
  window.location.href = "/signin";
}
