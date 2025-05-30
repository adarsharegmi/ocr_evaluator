import re
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load the OCR report
with open("ocr_comparison_report.txt", "r", encoding="utf-8") as file:
    report = file.read()

# Function to extract metric blocks
def extract_engine_metrics(report_text):
    engines = ["Azure Document Intelligence", "AWS Textract", "Tesseract", "EasyOCR"]
    data = []
    for engine in engines:
        pattern = rf"{engine}:\s*[\s\S]*?Processing Time: ([\d.]+)s[\s\S]*?Text Length: (\d+) characters[\s\S]*?Word Count: (\d+) words"
        conf_match = re.search(rf"{engine}:[\s\S]*?Confidence: ([\d.]+|None)", report_text)
        match = re.search(pattern, report_text)
        if match:
            processing_time = float(match.group(1))
            text_length = int(match.group(2))
            word_count = int(match.group(3))
        else:
            processing_time, text_length, word_count = None, 0, 0
        confidence = float(conf_match.group(1)) if conf_match and conf_match.group(1) != "None" else None
        data.append({
            "Engine": engine.replace("Azure Document Intelligence", "Azure").replace("AWS Textract", "Textract"),
            "text_length": text_length,
            "word_count": word_count,
            "processing_time": processing_time,
            "confidence": confidence
        })
    return pd.DataFrame(data)

def extract_similarity_metrics(report_text):
    sim_pattern = r"([A-Za-z]+)_vs_([A-Za-z]+):\s+character_similarity: ([\d.]+)\s+word_similarity: ([\d.]+)\s+edit_distance: (\d+)"
    similarities = {}
    for match in re.finditer(sim_pattern, report_text):
        a, b = match.group(1), match.group(2)
        char_sim = float(match.group(3))
        word_sim = float(match.group(4))
        edit_dist = int(match.group(5))
        similarities[(a, b)] = [char_sim, word_sim, edit_dist]
    return similarities



def compute_scores(metrics_df, similarity_data):
    scores = {}
    max_edit = max([v[2] for v in similarity_data.values()] or [1])
    max_word_count = metrics_df["word_count"].max() or 1

    for engine in metrics_df["Engine"]:
        char_sim_score = 0
        word_sim_score = 0
        edit_penalty = 0

        for (a, b), (char_sim, word_sim, edit_dist) in similarity_data.items():
            if engine in (a, b):
                char_sim_score += char_sim
                word_sim_score += word_sim
                edit_penalty += edit_dist / max_edit  # normalized

        # Average over comparison pairs
        n_pairs = sum(1 for key in similarity_data if engine in key)
        if n_pairs > 0:
            char_sim_score /= n_pairs
            word_sim_score /= n_pairs
            edit_penalty /= n_pairs

        row = metrics_df[metrics_df["Engine"] == engine].iloc[0]
        confidence = row["confidence"] if pd.notnull(row["confidence"]) else 0.5
        word_bonus = row["word_count"] / max_word_count

        # Weighted scoring formula
        score = (
            char_sim_score * 0.4 +
            word_sim_score * 0.3 +
            word_bonus * 0.2 +
            confidence * 0.1 -
            edit_penalty * 0.3  # penalty
        )

        scores[engine] = round(score, 4)

    return scores



# Extract metrics
ocr_metrics = extract_engine_metrics(report)
similarity_data = extract_similarity_metrics(report)
engines = ocr_metrics["Engine"].tolist()


# Compute scores
scores = compute_scores(ocr_metrics, similarity_data)

# Convert to DataFrame for plotting
scores_df = pd.DataFrame([
    {"Engine": engine, "Score": score}
    for engine, score in scores.items()
])

# Sort for visualization
scores_df.sort_values("Score", ascending=False, inplace=True)

# Plot
plt.figure(figsize=(8, 5))
sns.barplot(data=scores_df, x="Score", y="Engine", palette="viridis")

plt.title("OCR Engine Ranking by Composite Score")
plt.xlabel("Score")
plt.ylabel("OCR Engine")
plt.xlim(0, 1)  # Scores are normalized between 0 and 1
plt.grid(axis="x", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()
plt.savefig("ocr_engine_ranking.png")  # Save the plot as an image


# another plot visualization


# Bar plots
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
sns.barplot(data=ocr_metrics, x="Engine", y="text_length", ax=axs[0, 0])
axs[0, 0].set_title("Text Length")

sns.barplot(data=ocr_metrics, x="Engine", y="word_count", ax=axs[0, 1])
axs[0, 1].set_title("Word Count")

sns.barplot(data=ocr_metrics, x="Engine", y="processing_time", ax=axs[1, 0])
axs[1, 0].set_title("Processing Time (s)")

sns.barplot(data=ocr_metrics.dropna(), x="Engine", y="confidence", ax=axs[1, 1])
axs[1, 1].set_title("Confidence")

plt.tight_layout()
# plt.show()
# exprt to image 
plt.savefig("ocr_metrics.png")

# Initialize similarity matrices
char_sim = pd.DataFrame(index=engines, columns=engines, dtype=float)
word_sim = pd.DataFrame(index=engines, columns=engines, dtype=float)
edit_dist = pd.DataFrame(index=engines, columns=engines, dtype=float)

for (a, b), (char, word, edit) in similarity_data.items():
    if a not in engines or b not in engines:
        continue
    char_sim.loc[a, b] = char
    char_sim.loc[b, a] = char
    word_sim.loc[a, b] = word
    word_sim.loc[b, a] = word
    edit_dist.loc[a, b] = edit
    edit_dist.loc[b, a] = edit

for df, fill in zip([char_sim, word_sim], [1.0, 1.0]):
    for e in engines:
        df.loc[e, e] = fill

for e in engines:
    edit_dist.loc[e, e] = 0

# Heatmaps
fig, axs = plt.subplots(1, 3, figsize=(18, 5))
sns.heatmap(char_sim, annot=True, cmap="Blues", ax=axs[0])
axs[0].set_title("Character Similarity")

sns.heatmap(word_sim, annot=True, cmap="Greens", ax=axs[1])
axs[1].set_title("Word Similarity")

sns.heatmap(edit_dist, annot=True, cmap="Reds", ax=axs[2])
axs[2].set_title("Edit Distance")

plt.tight_layout()
plt.show()
