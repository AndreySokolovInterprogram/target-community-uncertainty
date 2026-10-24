# LLM prompts

Source: `src/jise/llm_ollama.py`. Models: `qwen2.5:7b-instruct`, `qwen3:8b` (Ollama, chat API).
Decoding: temperature 0, seed 0, one output token (`num_predict = 1`), `think = false`;
P(class 1) is computed from the top-10 token log-probabilities of `0` and `1`.
The group text (title + description, query-masked variant in the paper) is truncated to 1,500 characters.

The prompts are in Russian (as used in the experiments); English translations below are ours and were not
given to the models.

## System message (zero-shot and few-shot)

Original:

```
Ты классифицируешь сообщества ВКонтакте. Ответь одной цифрой: 1 — если сообщество является службой доставки готовой еды (суши, пицца, обеды, кафе или ресторан с доставкой и т.п.), 0 — если нет. Без пояснений.
```

Translation: *You classify VKontakte communities. Answer with a single digit: 1 if the community is a ready-food
delivery service (sushi, pizza, lunches, a café or restaurant with delivery, etc.), 0 if not. No explanations.*

## User message (one per group)

Original (`{text}` = group text, first 1,500 characters):

```
Сообщество:
{text}

Ответ (0 или 1):
```

Translation: *Community: {text} — Answer (0 or 1):*

## Zero-shot

```
system:    <system message>
user:      <user message for the test group>
```

## Few-shot ("4-shot" = 4 examples per class, 8 in total)

Examples are drawn at random from the training part of the current outer fold only
(`numpy.random.default_rng(100 + fold)`, 4 of class 0 and 4 of class 1, then shuffled). Each example is a
user/assistant pair whose answer is the gold label:

```
system:    <system message>
user:      <user message for example 1>
assistant: 0 | 1
...                                   (8 pairs)
user:      <user message for example 8>
assistant: 0 | 1
user:      <user message for the test group>
```

The indices of the examples used in each fold are stored in the per-fold prediction files (`shots_idx`), which are not included in this repository.
