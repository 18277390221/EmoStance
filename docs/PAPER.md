# Paper information

## Title

EmoStance: Response-Side Affective-Orientation Control for Empathetic Response Generation via Emoji Weak Supervision

## Authors

- Ziyuan Jin — ShanghaiTech University — `jinzy2024@shanghaitech.edu.cn`
- Yuxuan Ge — ShanghaiTech University — `geyx2023@shanghaitech.edu.cn`
- Zheng Tian — ShanghaiTech University — `tianzheng@shanghaitech.edu.cn` — corresponding author

## Abstract

Empathetic response generation requires models to decide not only what to say, but also how to respond to the previous speaker's affective situation. We formulate this as response-side affective-orientation control and use multi-annotator emoji distributions as weak affective–attitudinal evidence, rather than as output symbols or gold labels, to induce a latent control space that operationally approximates listener stance. We construct EmojiDialogue, an utterance-level extension of EmpatheticDialogues with emoji votes and confidence scores, and propose EmoStance, which models source-side affective expression, predicts a soft response-side orientation from dialogue context and speaker roles, and steers a frozen instruction-tuned LLM through continuous prefix embeddings. In blind pairwise evaluation with 20 annotators and 800 judgments, EmoStance achieves a 62.2% decisive win rate, with the clearest gains in contextual specificity and perceived responsiveness, while remaining complementary to external-knowledge methods.

The repository includes the [full paper PDF](../paper/EmoStance.pdf) and the [method figure](../assets/method_overview.pdf).
