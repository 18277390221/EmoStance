from __future__ import annotations

import argparse
import json
from pathlib import Path

from audit_lib import read_json


OVERLAY_CSS = """
<style>
.emoji-btn.llm-vote {
  border-color: #d97706 !important;
  border-radius: 999px !important;
  box-shadow: 0 0 0 3px rgba(217, 119, 6, .32) inset !important;
}
.emoji-btn.llm-top1 {
  border-color: #b91c1c !important;
  box-shadow:
    0 0 0 3px rgba(185, 28, 28, .42) inset,
    0 0 0 2px rgba(185, 28, 28, .55) !important;
}
.private-model-panel {
  margin-top: 10px;
  padding: 10px 12px;
  border: 1px solid #f3b37a;
  border-radius: 8px;
  background: #fff8ed;
  color: #4a2c0a;
}
.private-model-panel strong { color: #7f1d1d; }
.private-model-panel .model-chip {
  display: inline-block;
  margin: 4px 6px 0 0;
  padding: 3px 7px;
  border: 1px solid #f1b16d;
  border-radius: 999px;
  background: #fff;
}
</style>
"""


OVERLAY_JS = r"""
<script>
const MODEL_ANNOTATIONS = JSON.parse(document.getElementById("model-annotation-data").textContent);

function decorateModelAnnotations() {
  const item = ITEMS[current];
  const model = MODEL_ANNOTATIONS[item.sample_id];
  if (!model) return;
  const votes = model.llm_annotations || [];
  const voteEmoji = new Set(votes.map(v => v.selected_emoji));
  const voteText = votes.map(v => `${v.model}: ${v.selected_emoji} (${v.confidence}/5)`).join(" | ");
  document.querySelectorAll("#emoji-grid .emoji-btn").forEach(button => {
    const emoji = button.textContent;
    button.classList.toggle("llm-vote", voteEmoji.has(emoji));
    button.classList.toggle("llm-top1", emoji === model.top1_llm_emoji);
    if (voteEmoji.has(emoji)) {
      button.title = `${button.title || ""}  LLM vote: ${voteText}`.trim();
    }
  });
  let panel = document.getElementById("private-model-panel");
  if (!panel) {
    panel = document.createElement("div");
    panel.id = "private-model-panel";
    panel.className = "private-model-panel small";
    document.getElementById("selected-info").after(panel);
  }
  const chips = votes.map(v => `<span class="model-chip">${escapeHtml(v.model)} ${escapeHtml(v.selected_emoji)} ${escapeHtml(v.confidence)}/5</span>`).join("");
  panel.innerHTML = `<strong>Private model overlay.</strong> Circled emojis are LLM-selected emojis; red double circle is confidence-weighted top-1: <strong>${escapeHtml(model.top1_llm_emoji)}</strong><br>${chips}`;
}

const originalRenderItemForModelOverlay = renderItem;
renderItem = function() {
  originalRenderItemForModelOverlay();
  decorateModelAnnotations();
};
document.addEventListener("DOMContentLoaded", decorateModelAnnotations);
</script>
"""


def json_script(tag_id: str, payload: object) -> str:
    text = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    return f'<script id="{tag_id}" type="application/json">{text}</script>'


def build_model_payload(experiment_dir: Path) -> dict[str, dict[str, object]]:
    private_items = read_json(experiment_dir / "data/sampled_items_private.json")
    payload: dict[str, dict[str, object]] = {}
    for item in private_items:
        payload[item["sample_id"]] = {
            "top1_llm_emoji": item["top1_llm_emoji"],
            "llm_annotations": item["llm_annotations"],
            "llm_qE": item["llm_qE"],
        }
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a private reviewer copy with LLM emoji annotations circled.")
    parser.add_argument("--experiment-dir", default="human_llm_emoji_audit")
    parser.add_argument("--annotator-html", default="human_llm_emoji_audit/html/annotator_1.html")
    parser.add_argument("--output-html", default="human_llm_emoji_audit/reports/annotator_1_model_overlay_private.html")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_dir = Path(args.experiment_dir)
    source = Path(args.annotator_html)
    output = Path(args.output_html)
    html_text = source.read_text(encoding="utf-8")
    if "model-annotation-data" in html_text:
        raise SystemExit(f"{source} already appears to contain a model overlay")
    model_payload = build_model_payload(experiment_dir)
    html_text = html_text.replace("</head>", OVERLAY_CSS + "\n</head>")
    html_text = html_text.replace(
        "</body>",
        json_script("model-annotation-data", model_payload) + "\n" + OVERLAY_JS + "\n</body>",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html_text, encoding="utf-8")
    print(f"Wrote private model-overlay HTML: {output}")
    print("Do not give this private overlay file to human annotators.")


if __name__ == "__main__":
    main()
