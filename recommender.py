# ============================================================
# recommender.py — The core recommendation engine
#
# What this file does:
# 1. Loads your CSV movie dataset
# 2. Extracts genres, languages, countries dynamically
# 3. Builds semantic embeddings using Sentence Transformers
# 4. Stores embeddings in ChromaDB (local vector database)
# 5. Searches by semantic similarity + hard filters
# 6. Returns ranked results with confidence scores
#
# No external API needed. Runs fully offline after setup.
# ============================================================

import os
import re
import ast
import pandas as pd
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# ── Constants — edit these to match your CSV column names ───
# If your CSV has different column names, change them here.
COL_TITLE    = "title"           # movie title
COL_OVERVIEW = "overview"        # plot / description
COL_GENRES   = "genres"          # genres (pipe-separated or list)
COL_LANGUAGE = "original_language"  # language code or name
COL_COUNTRY  = "production_countries"  # country of production
COL_ACTORS   = "cast"            # actors (optional)
COL_KEYWORDS = "keywords"        # keywords / themes (optional)
COL_YEAR     = "release_date"    # release year or full date
COL_RATING   = "vote_average"    # IMDb-style score
COL_FORMAT   = "type"            # "movie" or "series" (optional)

# Path where ChromaDB stores its local vector database files
CHROMA_DIR   = "./chroma_db"
COLLECTION   = "movifinds_movies"

# Embedding model — runs locally, downloads once (~90MB)
# "all-MiniLM-L6-v2" is fast and accurate for this use case
EMBED_MODEL  = "all-MiniLM-L6-v2"


class MovieRecommender:
    """
    Main recommendation engine.
    Call search() to get recommendations.
    """

    def __init__(self, dataset_path: str):
        print(f"📂 Loading dataset from: {dataset_path}")
        self.df = self._load_dataset(dataset_path)
        print(f"✅ Loaded {len(self.df)} movies")

        print("🧠 Loading embedding model (downloads once ~90MB)...")
        self.model = SentenceTransformer(EMBED_MODEL)
        print("✅ Embedding model ready")

        print("🗄️  Setting up vector database...")
        self.client     = chromadb.PersistentClient(path=CHROMA_DIR)
        self.collection = self._setup_collection()
        print("✅ Vector database ready")


    # ── DATASET LOADING ──────────────────────────────────────
    def _load_dataset(self, path: str) -> pd.DataFrame:
        """
        Loads CSV and normalises all important columns.
        Handles messy data gracefully.
        """
        df = pd.read_csv(path, low_memory=False)

        # Fill missing text fields with empty string
        for col in [COL_TITLE, COL_OVERVIEW, COL_GENRES,
                    COL_LANGUAGE, COL_COUNTRY, COL_ACTORS, COL_KEYWORDS]:
            if col in df.columns:
                df[col] = df[col].fillna("").astype(str)

        # Parse genres into a clean list per row
        df["_genres_list"] = df[COL_GENRES].apply(self._parse_list_field)

        # Parse countries into a clean list per row
        df["_countries_list"] = df[COL_COUNTRY].apply(self._parse_list_field) \
            if COL_COUNTRY in df.columns else [[]]

        # Normalise language to readable name
        df["_language"] = df[COL_LANGUAGE].apply(self._normalise_language) \
            if COL_LANGUAGE in df.columns else ""

        # Extract year from date
        df["_year"] = df[COL_YEAR].apply(self._extract_year) \
            if COL_YEAR in df.columns else 0

        # Normalise rating to 0-10
        df["_rating"] = pd.to_numeric(df[COL_RATING], errors="coerce").fillna(0) \
            if COL_RATING in df.columns else 0

        # Build the combined text used for embeddings
        df["_combined_text"] = df.apply(self._build_combined_text, axis=1)

        # Drop rows with no title or no overview (not useful)
        df = df[df[COL_TITLE].str.strip() != ""]
        df = df.reset_index(drop=True)

        return df


    def _parse_list_field(self, value: str) -> list:
        """
        Handles multiple formats genres/countries can appear in:
        - Pipe separated:  "Action|Comedy|Drama"
        - JSON list:       [{"name": "Action"}, {"name": "Comedy"}]
        - Comma separated: "Action, Comedy, Drama"
        - Plain string:    "Action"
        """
        if not value or value == "nan":
            return []
        value = value.strip()

        # Try JSON/dict list format (TMDb style)
        if value.startswith("["):
            try:
                parsed = ast.literal_eval(value)
                if isinstance(parsed, list):
                    result = []
                    for item in parsed:
                        if isinstance(item, dict):
                            result.append(item.get("name", ""))
                        elif isinstance(item, str):
                            result.append(item)
                    return [r.strip() for r in result if r.strip()]
            except Exception:
                pass

        # Pipe separated
        if "|" in value:
            return [v.strip() for v in value.split("|") if v.strip()]

        # Comma separated
        if "," in value:
            return [v.strip() for v in value.split(",") if v.strip()]

        # Plain string
        return [value.strip()] if value.strip() else []


    def _normalise_language(self, code: str) -> str:
        """Converts language codes like 'en' to 'English'."""
        lang_map = {
            "en": "English", "fr": "French", "es": "Spanish",
            "de": "German",  "it": "Italian", "pt": "Portuguese",
            "ja": "Japanese", "ko": "Korean", "zh": "Chinese",
            "hi": "Hindi",   "ar": "Arabic",  "ru": "Russian",
            "tr": "Turkish", "nl": "Dutch",   "sv": "Swedish",
            "pl": "Polish",  "da": "Danish",  "fi": "Finnish",
            "no": "Norwegian", "th": "Thai",  "id": "Indonesian",
            "ms": "Malay",   "vi": "Vietnamese", "fa": "Persian",
            "he": "Hebrew",  "cs": "Czech",   "hu": "Hungarian",
            "ro": "Romanian", "uk": "Ukrainian", "bn": "Bengali",
            "ta": "Tamil",   "te": "Telugu",  "ml": "Malayalam",
            "yo": "Yoruba",  "ig": "Igbo",    "ha": "Hausa",
        }
        code = str(code).strip().lower()
        return lang_map.get(code, code.capitalize())


    def _extract_year(self, value) -> int:
        """Extracts 4-digit year from strings like '2019-07-04'."""
        match = re.search(r"\b(19|20)\d{2}\b", str(value))
        return int(match.group()) if match else 0


    def _build_combined_text(self, row) -> str:
        """
        Combines multiple fields into one rich text string
        used to generate the movie's embedding vector.
        The more fields included, the better the search quality.
        """
        parts = []

        title    = str(row.get(COL_TITLE, "")).strip()
        overview = str(row.get(COL_OVERVIEW, "")).strip()
        genres   = " ".join(row.get("_genres_list", []))
        actors   = str(row.get(COL_ACTORS, "")).strip() if COL_ACTORS in row.index else ""
        keywords = str(row.get(COL_KEYWORDS, "")).strip() if COL_KEYWORDS in row.index else ""
        language = str(row.get("_language", "")).strip()
        country  = " ".join(row.get("_countries_list", []))

        # Weight important fields by repeating them
        if title:    parts.append(f"Title: {title}. {title}.")
        if genres:   parts.append(f"Genres: {genres}. {genres}.")
        if overview: parts.append(f"Overview: {overview}")
        if keywords: parts.append(f"Themes: {keywords}")
        if actors:   parts.append(f"Cast: {actors}")
        if language: parts.append(f"Language: {language}")
        if country:  parts.append(f"Country: {country}")

        return " ".join(parts)


    # ── VECTOR DATABASE SETUP ────────────────────────────────
    def _setup_collection(self):
        """
        Creates or loads the ChromaDB collection.
        If movies are already embedded (from a previous run),
        skips re-embedding — making restarts very fast.
        """
        try:
            collection = self.client.get_collection(COLLECTION)
            existing   = collection.count()
            if existing >= len(self.df) * 0.9:
                print(f"⚡ Using cached embeddings ({existing} movies)")
                return collection
            else:
                print("🔄 Dataset changed — rebuilding embeddings...")
                self.client.delete_collection(COLLECTION)
        except Exception:
            pass

        # Create fresh collection
        collection = self.client.create_collection(
            name=COLLECTION,
            metadata={"hnsw:space": "cosine"}   # cosine similarity
        )

        # Embed and store in batches (avoids memory issues)
        batch_size = 100
        total      = len(self.df)
        print(f"⚙️  Embedding {total} movies (this runs once, then cached)...")

        for start in range(0, total, batch_size):
            end   = min(start + batch_size, total)
            batch = self.df.iloc[start:end]

            texts = batch["_combined_text"].tolist()
            ids   = [str(i) for i in range(start, end)]

            # Generate embeddings for this batch
            embeddings = self.model.encode(texts, show_progress_bar=False).tolist()

            # Build metadata for filtering later
            metadatas = []
            for _, row in batch.iterrows():
                metadatas.append({
                    "title":     str(row.get(COL_TITLE, "")),
                    "year":      int(row.get("_year", 0)),
                    "rating":    float(row.get("_rating", 0)),
                    "genres":    "|".join(row.get("_genres_list", [])),
                    "language":  str(row.get("_language", "")),
                    "countries": "|".join(row.get("_countries_list", [])),
                    "overview":  str(row.get(COL_OVERVIEW, ""))[:500],
                })

            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas
            )

            pct = int((end / total) * 100)
            print(f"   {pct}% — embedded {end}/{total} movies", end="\r")

        print(f"\n✅ All {total} movies embedded and stored")
        return collection


    # ── FILTER OPTIONS ───────────────────────────────────────
    def get_filter_options(self) -> dict:
        """
        Dynamically extracts all unique genres, languages,
        and countries from the dataset.
        Frontend calls this on page load.
        """
        genres    = sorted(set(
            g for sublist in self.df["_genres_list"] for g in sublist if g
        ))
        languages = sorted(set(
            l for l in self.df["_language"] if l and l != "nan"
        ))
        countries = sorted(set(
            c for sublist in self.df["_countries_list"] for c in sublist if c
        ))

        return {
            "genres":    genres,
            "languages": languages,
            "countries": countries,
            "total_movies": 5000
        }


    # ── QUERY BUILDER ────────────────────────────────────────
    def build_query(self, genres, mood, situation, energy,
                    ending, format_type, audience,
                    world_val, custom_text, similar_to) -> str:
        """
        Converts all user selections into a rich natural-language
        query string that will be embedded and searched.
        """
        parts = []

        if genres:
            parts.append(f"A {' and '.join(genres)} film.")

        mood_map = {
            "suspenseful": "full of suspense and tension",
            "feel-good":   "uplifting and feel-good",
            "emotional":   "deeply emotional and moving",
            "mind-bending":"mind-bending and thought-provoking",
            "dark and gritty": "dark, gritty, and intense",
            "light-hearted":   "light-hearted and fun",
            "intense":     "gripping and intense"
        }
        if mood:
            mood_desc = " and ".join(mood_map.get(m, m) for m in mood)
            parts.append(f"The tone is {mood_desc}.")

        if situation:
            parts.append(f"Good for {' or '.join(situation)}.")

        energy_map = {
            "fast-paced":        "fast-paced with lots of action",
            "slow-burn":         "slow-burn with gradual tension",
            "thrilling":         "thrilling throughout",
            "relaxing":          "relaxing and easy to watch",
            "thought-provoking": "makes you think deeply"
        }
        if energy:
            energy_desc = " and ".join(energy_map.get(e, e) for e in energy)
            parts.append(f"Pacing: {energy_desc}.")

        if ending:
            parts.append(f"I prefer a {' or '.join(ending)}.")

        if format_type:
            parts.append(f"Format: {format_type}.")

        if audience:
            parts.append(f"Suitable for {audience} viewers.")

        if world_val < 30:
            parts.append("Very realistic, no fantasy or supernatural elements.")
        elif world_val > 70:
            parts.append("Fantasy, supernatural, or sci-fi world-building.")
        else:
            parts.append("Mix of realistic and fantastical elements.")

        if similar_to:
            parts.append(f"Similar to {similar_to}.")

        if custom_text:
            parts.append(custom_text)

        # Fallback if nothing selected
        if not parts:
            return "popular highly rated movie"

        return " ".join(parts)


    # ── MAIN SEARCH ──────────────────────────────────────────
    def search(self, query: str, genres: list, language: str,
               country: str, top_k: int = 10) -> list:
        """
        1. Embeds the query
        2. Searches ChromaDB for similar movies
        3. Applies genre / language / country hard filters
        4. Ranks by combined similarity + rating score
        5. Returns top_k results
        """

        # Handle empty query gracefully
        if not query.strip():
            query = "popular highly rated movie"

        # Embed the query
        query_embedding = self.model.encode([query])[0].tolist()

        # Fetch more than needed so filtering doesn't leave us with too few
        fetch_k = min(top_k * 8, len(self.df))

        # Search vector database
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=fetch_k,
            include=["metadatas", "distances"]
        )

        metadatas = results["metadatas"][0]
        distances = results["distances"][0]   # lower = more similar (cosine)

        # Convert distance to similarity score 0-1
        similarities = [max(0.0, 1.0 - d) for d in distances]

        # Apply hard filters + score each result
        filtered = []
        for meta, sim in zip(metadatas, similarities):
            movie_genres    = meta.get("genres", "").split("|")
            movie_language  = meta.get("language", "")
            movie_countries = meta.get("countries", "").split("|")
            movie_rating    = float(meta.get("rating", 0))

            # Genre filter — movie must contain AT LEAST ONE selected genre
            if genres:
                genre_match = any(
                    g.lower() in [mg.lower() for mg in movie_genres]
                    for g in genres
                )
                if not genre_match:
                    continue

            # Language filter — exact match if selected
            if language and language.lower() not in movie_language.lower():
                continue

            # Country filter — movie must be from that country
            if country:
                country_match = any(
                    country.lower() in mc.lower()
                    for mc in movie_countries
                )
                if not country_match:
                    continue

            # Genre overlap score — rewards movies matching MORE selected genres
            genre_overlap = 0
            if genres:
                matched = sum(
                    1 for g in genres
                    if g.lower() in [mg.lower() for mg in movie_genres]
                )
                genre_overlap = matched / len(genres)

            # Combined score: 60% semantic + 30% genre overlap + 10% rating
            rating_norm   = min(movie_rating / 10.0, 1.0)
            combined_score = (0.60 * sim) + (0.30 * genre_overlap) + (0.10 * rating_norm)

            # Confidence label
            if combined_score > 0.75:
                confidence = "Excellent match"
            elif combined_score > 0.55:
                confidence = "Great match"
            elif combined_score > 0.40:
                confidence = "Good match"
            else:
                confidence = "Possible match"

            filtered.append({
                "title":       meta.get("title", "Unknown"),
                "year":        meta.get("year", 0),
                "score":       round(movie_rating, 1),
                "genre":       ", ".join(movie_genres[:3]),
                "country":     movie_countries[0] if movie_countries else "",
                "language":    movie_language,
                "description": meta.get("overview", ""),
                "tags":        movie_genres[:4],
                "similarity":  round(sim * 100, 1),
                "confidence":  confidence,
                "_sort_score": combined_score
            })

        # Sort by combined score descending
        filtered.sort(key=lambda x: x["_sort_score"], reverse=True)

        # Fallback: if filters removed everything, return top semantic matches
        if not filtered:
            filtered = self._fallback_results(similarities, metadatas, top_k)

        # Add rank and clean up internal fields
        output = []
        for i, movie in enumerate(filtered[:top_k]):
            movie.pop("_sort_score", None)
            movie["rank"] = i + 1
            output.append(movie)

        return output


    # ── SIMILAR MOVIES ───────────────────────────────────────
    def find_similar(self, title: str, top_k: int = 5) -> list:
        """
        Finds movies most similar to a given title
        by looking up its embedding and searching neighbours.
        """
        # Find the movie in the dataset
        match = self.df[self.df[COL_TITLE].str.lower() == title.lower()]
        if match.empty:
            # Try partial match
            match = self.df[self.df[COL_TITLE].str.lower().str.contains(
                title.lower(), na=False
            )]

        if match.empty:
            return []

        # Use the first match's combined text as the query
        query_text = match.iloc[0]["_combined_text"]
        query_emb  = self.model.encode([query_text])[0].tolist()

        results = self.collection.query(
            query_embeddings=[query_emb],
            n_results=top_k + 1,   # +1 because the movie itself will appear
            include=["metadatas", "distances"]
        )

        metadatas  = results["metadatas"][0]
        distances  = results["distances"][0]
        output     = []

        for meta, dist in zip(metadatas, distances):
            # Skip the movie itself
            if meta.get("title", "").lower() == title.lower():
                continue
            sim = max(0.0, 1.0 - dist)
            output.append({
                "rank":        len(output) + 1,
                "title":       meta.get("title", ""),
                "year":        meta.get("year", 0),
                "score":       round(float(meta.get("rating", 0)), 1),
                "genre":       meta.get("genres", "").replace("|", ", "),
                "description": meta.get("overview", ""),
                "similarity":  round(sim * 100, 1),
                "confidence":  "Similar movie"
            })
            if len(output) >= top_k:
                break

        return output


    # ── FALLBACK ─────────────────────────────────────────────
    def _fallback_results(self, similarities, metadatas, top_k) -> list:
        """
        When hard filters remove all results, return the
        top semantic matches regardless of filter constraints.
        """
        combined = sorted(
            zip(similarities, metadatas),
            key=lambda x: x[0],
            reverse=True
        )[:top_k]

        output = []
        for sim, meta in combined:
            genres = meta.get("genres", "").split("|")
            output.append({
                "title":       meta.get("title", "Unknown"),
                "year":        meta.get("year", 0),
                "score":       round(float(meta.get("rating", 0)), 1),
                "genre":       ", ".join(genres[:3]),
                "country":     meta.get("countries", "").split("|")[0],
                "language":    meta.get("language", ""),
                "description": meta.get("overview", ""),
                "tags":        genres[:4],
                "similarity":  round(sim * 100, 1),
                "confidence":  "Broad match — filters relaxed",
                "_sort_score": sim
            })
        return output
