from flask import Flask, request, jsonify
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load once, CPU only
model = SentenceTransformer(
    "paraphrase-MiniLM-L6-v2",
    device="cpu"
)


@app.route("/compare_issues", methods=["POST"])
def compare_issues():
    data = request.get_json(force=True)

    issue1_text = f"{data.get('issue1_title', '')} {
        data.get('issue1_body', '')}"
    issue2_text = f"{data.get('issue2_title', '')} {
        data.get('issue2_body', '')}"

    # Use numpy, NOT torch tensors
    embeddings = model.encode(
        [issue1_text, issue2_text],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    similarity = float(
        cosine_similarity(
            embeddings[0].reshape(1, -1),
            embeddings[1].reshape(1, -1)
        )[0][0]
    )

    return jsonify({"similarity": similarity}), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8233))
    app.run(host="0.0.0.0", port=port)
