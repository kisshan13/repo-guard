import os
import re

from flask import Flask, jsonify, request
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# BAAI/bge-m3 for state-of-the-art accuracy
model = SentenceTransformer("BAAI/bge-m3", device="cpu")


def clean(text: str) -> str:
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def issue_text(title: str, body: str) -> str:
    # Combine title and body, handling empty body case
    title_clean = clean(title) if title else ""
    body_clean = clean(body) if body else ""

    # If body is empty or minimal, rely more on title
    if not body_clean or body_clean == "...":
        return f"GitHub Issue: {title_clean}"

    return f"GitHub Issue: {title_clean}. {body_clean}"


@app.route("/compare_issues", methods=["POST"])
def compare_issues():
    data = request.get_json(force=True)
    issue1_text = issue_text(
        data.get("issue1_title", ""),
        data.get("issue1_body", ""),
    )
    issue2_text = issue_text(
        data.get("issue2_title", ""),
        data.get("issue2_body", ""),
    )

    # BGE-M3 supports passage encoding
    embeddings = model.encode(
        [issue1_text, issue2_text],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    similarity = float(
        cosine_similarity(
            embeddings[0].reshape(1, -1),
            embeddings[1].reshape(1, -1),
        )[0][0]
    )

    # Adjusted thresholds based on real-world GitHub issue comparison
    # These should be tuned based on your specific dataset
    label = (
        "duplicate"
        if similarity >= 0.75
        else "possible"
        if similarity >= 0.65
        else "different"
    )

    return jsonify(
        {
            "similarity": similarity,
            "label": label,
        }
    ), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8333))
    app.run(host="0.0.0.0", port=port)
