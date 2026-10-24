# Class definition and annotation instructions

English translation of the instruction sheet given to the annotator (the original was in Russian;
see `src/jise/make_annotation.py`).

## What is checked

Group labels were not assigned manually; they come from the VK search query that returned the group:

* **delivery** — the group was found by the queries "food delivery", "pizza", "sushi", "burgers", "rolls",
  "food to your home", "lunch delivery";
* **other** — by the queries "news", "humour", "memes", "photography", "sport", "books".

The task is to judge whether this label is correct.

## What counts as "delivery"

The community is a **ready-food delivery service**: it sells and delivers (or sells for take-away) ready-made
food — sushi, pizza, lunches, shashlik, baked goods, etc. This also includes cafés and restaurants if the group
offers delivery or ordering of ready food, and chains / franchises (e.g. a pizza chain in a specific city).

**Not "delivery":** cooking blogs and recipes, mukbangs, food reviews, grocery shops without ready food,
aggregators, courier job offers, memes and "about food" pages, and anything not related to food.

## How to annotate

For each row read the title and the description; open the link if needed. In the column `label_correct` choose:

* **yes** — the label in the column `label` matches the definition above;
* **no** — the label is wrong (e.g. "delivery" for a recipe blog, or "other" for a delivery service);
* **unclear** — it cannot be decided from the title, description and page (group deleted, closed, empty, etc.).

The column `comment` may briefly explain disputable cases.

Judge the community, not how the search works. Do not correct the label; only mark yes / no / unclear.

Sample: stratified random, 150 "delivery" + 150 "other", seed 20261020; rows shuffled.
