# ============================================================
# app.py — MoviFinds Local Recommendation Engine
# Flask server that handles all routes.
# No external AI API needed — fully offline capable.
# ============================================================

from flask import Flask, render_template, request, jsonify
from recommender import MovieRecommender
import os

app = Flask(__name__)

# ── Boot the recommender once when the server starts ────────
# This loads the dataset, builds embeddings, and sets up
# the vector database. Takes ~30-60 seconds on first run.
print("🎬 Starting MoviFinds recommendation engine...")
recommender = MovieRecommender(dataset_path="data/movies.csv")
print("✅ Engine ready!")


# ── Route: Home page ────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


# ── Route: Get filter options from the dataset ───────────────
# Called on page load to dynamically populate dropdowns/chips
@app.route("/filters", methods=["GET"])
def get_filters():
    """
    Returns all unique genres, languages, and countries
    extracted directly from the dataset.
    Frontend uses this to build the filter UI dynamically.
    """
    try:
        filters = recommender.get_filter_options()
        return jsonify({"success": True, **filters})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ── Route: Get movie recommendations ────────────────────────
@app.route("/recommend", methods=["POST"])
def recommend():
    """
    Receives user filter selections + free-text query,
    runs semantic search + filter matching,
    and returns ranked movie recommendations.
    """
    try:
        data = request.get_json()

        # Extract user selections
        genres      = data.get("genre", [])        # list, up to 5
        language    = data.get("language", "")     # single string
        country     = data.get("country", "")      # single string
        mood        = data.get("mood", [])
        situation   = data.get("situation", [])
        energy      = data.get("energy", [])
        ending      = data.get("ending", [])
        format_type = data.get("format", "movie")
        audience    = data.get("audience", "")
        world_val   = int(data.get("world_slider", 50))
        custom_text = data.get("custom_text", "").strip()
        similar_to  = data.get("similar_to", "")  # for "similar movies" feature

        # Build the semantic query from all selections
        query = recommender.build_query(
            genres=genres, mood=mood, situation=situation,
            energy=energy, ending=ending, format_type=format_type,
            audience=audience, world_val=world_val,
            custom_text=custom_text, similar_to=similar_to
        )

        # Run retrieval + filtering + ranking
        movies = recommender.search(
            query=query,
            genres=genres,
            language=language,
            country=country,
            top_k=10
        )

        return jsonify({
            "success": True,
            "movies": movies,
            "query_used": query   # shown in prompt preview card
        })

    except Exception as e:
        print(f"❌ Recommendation error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# ── Route: Similar movies ────────────────────────────────────
@app.route("/similar", methods=["POST"])
def similar():
    """
    Given a movie title, find the most similar movies
    in the dataset using vector similarity.
    """
    try:
        data  = request.get_json()
        title = data.get("title", "")
        top_k = int(data.get("top_k", 5))

        movies = recommender.find_similar(title=title, top_k=top_k)
        return jsonify({"success": True, "movies": movies})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ── Start server ─────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=5000)
