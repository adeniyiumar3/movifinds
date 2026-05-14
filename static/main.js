// ============================================================
// main.js — MoviFinds LOCAL Frontend
//
// What this does:
// 1. On page load: fetches genres/languages/countries from
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
  energy:      [],      // up to 2
  ending:      [],      // up to 2
  language:    "",
  country:     "",
  format:      "movie",
  audience:    "Teen & Young Adult",
  worldSlider: 45,
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

    // Populate country dropdown
    buildSelect("country-select", data.countries);

    // Update dataset stats card
    document.getElementById("stat-genres").textContent    = data.genres.length;
    document.getElementById("stat-languages").textContent = data.languages.length;
    document.getElementById("stat-countries").textContent = data.countries.length;
    // Movie count comes from total (set by backend if available)
    if (data.total_movies) {
      document.getElementById("stat-movies").textContent = data.total_movies.toLocaleString();
    }

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


// ── WORLD SLIDER ─────────────────────────────────────────────
const sliderQuotes = [
  "Grounded in reality — no fantasy elements.",
  "Mostly realistic with a subtle magical touch.",
  "A balanced mix of realism and fantasy.",
  "Leans into fantasy with some real-world grounding.",
  "Full fantasy and supernatural — anything goes."
];
function updateSlider(v) {
  state.worldSlider = parseInt(v);
  const i = Math.min(4, Math.floor(v / 21));
  document.getElementById("slider-quote").textContent = `"${sliderQuotes[i]}"`;
  updatePrompt();
}


// ── LANGUAGE / COUNTRY dropdowns ─────────────────────────────
document.getElementById("language-select").addEventListener("change", function() {
  state.language = this.value;
  updatePrompt();
});
document.getElementById("country-select").addEventListener("change", function() {
  state.country = this.value;
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
  const { genre, mood, situation, energy, ending,
          language, country, format, audience,
          worldSlider, similarTo, customText } = state;

  const hasSelection = genre.length || mood.length || situation.length ||
                       energy.length || ending.length || similarTo || customText;
  if (!hasSelection) return null;

  const join = arr => arr.join(" and ");
  const parts = [];

  if (genre.length)     parts.push(`Genre: ${join(genre)}.`);
  if (mood.length)      parts.push(`Mood: ${join(mood)}.`);
  if (situation.length) parts.push(`For: ${join(situation)}.`);
  if (energy.length)    parts.push(`Vibe: ${join(energy)}.`);
  if (ending.length)    parts.push(`Ending: ${join(ending)}.`);
  if (language)         parts.push(`Language: ${language}.`);
  if (country)          parts.push(`Country: ${country}.`);
  if (format)           parts.push(`Format: ${format}.`);
  if (audience)         parts.push(`Audience: ${audience}.`);

  const worldDesc = worldSlider < 30 ? "realistic" :
                    worldSlider > 70 ? "fantasy/supernatural" : "mix of both";
  parts.push(`World-building: ${worldDesc}.`);

  if (similarTo)    parts.push(`Similar to: "${similarTo}".`);
  if (customText.trim()) parts.push(customText.trim());

  return parts.join(" ");
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
}


// ── CLEAR ALL ────────────────────────────────────────────────
function clearAll() {
  document.querySelectorAll(".chip").forEach(c => c.classList.remove("sel", "disabled"));
  document.querySelectorAll(".counter").forEach(c => {
    const max = c.id.includes("genre") ? "5" : "2";
    c.textContent = `0/${max}`;
    c.classList.remove("full");
  });
  ["genre","mood","situation","energy","ending"].forEach(k => state[k] = []);

  document.getElementById("world-slider").value    = 45;
  document.getElementById("slider-quote").textContent = '"Grounded with a hint of magic."';
  document.getElementById("custom-input").value    = "";
  document.getElementById("similar-input").value   = "";
  document.getElementById("language-select").value = "";
  document.getElementById("country-select").value  = "";

  state.worldSlider = 45;
  state.customText  = "";
  state.similarTo   = "";
  state.language    = "";
  state.country     = "";
  updatePrompt();
}


// ── GET RECOMMENDATIONS ──────────────────────────────────────
async function getRecommendations() {
  const btn         = document.getElementById("cta-btn");
  const btnText     = document.getElementById("cta-text");
  const placeholder = document.getElementById("results-placeholder");
  const shimmer     = document.getElementById("shimmer-list");
  const resultsList = document.getElementById("results-list");
  const label       = document.getElementById("results-label");

  // Show loading
  btn.disabled = true;
  btnText.textContent = "Searching...";
  placeholder.style.display = "none";
  shimmer.style.display     = "flex";
  shimmer.style.flexDirection = "column";
  resultsList.style.display = "none";
  resultsList.innerHTML     = "";
  label.textContent = "Searching your dataset...";

  document.getElementById("results-zone")
    .scrollIntoView({ behavior: "smooth", block: "start" });

  try {
    const payload = {
      genre:        state.genre,
      mood:         state.mood,
      situation:    state.situation,
      energy:       state.energy,
      ending:       state.ending,
      language:     state.language,
      country:      state.country,
      format:       state.format,
      audience:     state.audience,
      world_slider: state.worldSlider,
      similar_to:   state.similarTo,
      custom_text:  state.customText
    };

    const res  = await fetch("/recommend", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const data = await res.json();

    if (data.success) {
      label.textContent = `Top ${data.movies.length} results from your dataset`;
      renderMovies(data.movies);
    } else {
      showError("Error: " + (data.error || "Unknown error"));
    }

  } catch (err) {
    showError("Cannot connect to server. Is app.py running?");
  } finally {
    btn.disabled = false;
    btnText.textContent = "Get Recommendations";
    shimmer.style.display = "none";
  }
}


// ── RENDER MOVIE RESULTS ─────────────────────────────────────
function renderMovies(movies) {
  const list = document.getElementById("results-list");
  list.innerHTML = "";

  if (!movies.length) {
    list.innerHTML = `<div class="error-msg">No movies matched your filters. Try relaxing your language/country selection.</div>`;
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


// ── AVATAR DROPDOWN ──────────────────────────────────────────
function toggleAv() {
  document.getElementById("av-dd").classList.toggle("open");
}
document.addEventListener("click", e => {
  if (!e.target.closest("#avatar")) {
    const dd = document.getElementById("av-dd");
    if (dd) dd.classList.remove("open");
  }
});
