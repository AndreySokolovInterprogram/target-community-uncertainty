Manual check (one annotator, final version), 300 groups (150 delivery + 150 other, seed 20261020). Raw verdicts: {'yes': 278, 'no': 22}.

## Label noise (Wilson 95 %; overall = class-share weighted, interval = weighted Wilson bounds, conservative)

| class                     |   n_checked |   wrong |   share_wrong |   wilson_lo |   wilson_hi |   class_share_in_data |
|:--------------------------|------------:|--------:|--------------:|------------:|------------:|----------------------:|
| delivery                  |         150 |      22 |        0.1467 |      0.0989 |      0.2121 |                0.4871 |
| other                     |         150 |       0 |        0.0000 |      0.0000 |      0.0250 |                0.5129 |
| whole data set (weighted) |         300 |      22 |        0.0714 |      0.0482 |      0.1161 |                1.0000 |

## Wrong labels by category (our categories)

| category                                                        |   n |
|:----------------------------------------------------------------|----:|
| not food (shop, rental, flowers, etc.)                          |   5 |
| recipes / food blog / mukbang                                   |   5 |
| delivery-like by text, judged not a ready-food delivery service |   5 |
| food-related, not a ready-food service                          |   3 |
| promo codes / aggregator                                        |   3 |
| courier jobs                                                    |   1 |

|   № |     VK id | название                                                  | метка    | category                                                        |
|----:|----------:|:----------------------------------------------------------|:---------|:----------------------------------------------------------------|
|   7 | 199336561 | Комиссионка №1  l ЧЕРЕПОВЕЦ l ДОСТАВКА                    | доставка | not food (shop, rental, flowers, etc.)                          |
|   9 | 225001373 | ЕДА МУКБАНГ                                               | доставка | recipes / food blog / mukbang                                   |
|  19 |  28964242 | Капли, доставка горячих обедов на дом и в офис, кейтеринг | доставка | delivery-like by text, judged not a ready-food delivery service |
|  29 |   7362811 | На халяву еда вкуснее!!!!!!!!!!!!!!!!!                    | доставка | food-related, not a ready-food service                          |
|  47 | 228243090 | Твоя еда                                                  | доставка | recipes / food blog / mukbang                                   |
|  89 | 223875651 | Промокоды на доставку готовой еды суши, пицца             | доставка | promo codes / aggregator                                        |
|  92 |  51783845 | Доставка еды, продуктов Москва                            | доставка | promo codes / aggregator                                        |
|  96 | 220904171 | Курьеры Яндекс Еда                                        | доставка | courier jobs                                                    |
| 111 | 227473909 | Шарпи-пицца и шаурма                                      | доставка | delivery-like by text, judged not a ready-food delivery service |
| 119 | 223334262 | ПРОМОКОДЫ | ДОСТАВКА ЕДЫ                                  | доставка | promo codes / aggregator                                        |
| 148 |  40407167 | ДОМ НА БАЛИ                                               | доставка | not food (shop, rental, flowers, etc.)                          |
| 157 |  50629741 | Умная еда ED                                              | доставка | food-related, not a ready-food service                          |
| 177 | 226926417 | Домашняя еда                                              | доставка | delivery-like by text, judged not a ready-food delivery service |
| 205 | 142538177 | Полезная еда рецепты                                      | доставка | recipes / food blog / mukbang                                   |
| 218 | 119726958 | суши, роллы                                               | доставка | recipes / food blog / mukbang                                   |
| 230 |  78339402 | Сегодня Можно | Коктейли | Еда | Искусство                | доставка | food-related, not a ready-food service                          |
| 234 | 204878568 | StrelKa - Еда                                             | доставка | recipes / food blog / mukbang                                   |
| 271 | 105405623 | Доставка 24 Dostavka24.shop доставка еды в ЯНАО           | доставка | delivery-like by text, judged not a ready-food delivery service |
| 273 | 228303204 | ВАЗУЗА RELAX | ДОМ НА БЕРЕГУ | АРЕНДА ЛЕБЕДКИ             | доставка | not food (shop, rental, flowers, etc.)                          |
| 279 |  51581870 | "Вкусная жизнь" Доставка еды!                             | доставка | delivery-like by text, judged not a ready-food delivery service |
| 285 | 162728449 | Доставка цветов г.Пестово . «Формула Букета»              | доставка | not food (shop, rental, flowers, etc.)                          |
| 291 |  65719167 | коллажики за бургеры проект года                          | доставка | not food (shop, rental, flowers, etc.)                          |

## OOF predictions (query-masked) on the 300 groups, accuracy weighted to real class shares; 95 % CI = stratified bootstrap

| model                    | acc vs orig label    | acc vs corrected label   |   acc on all 11,612 (orig label) |
|:-------------------------|:---------------------|:-------------------------|---------------------------------:|
| e5-base fine-tuned, ens5 | 0.984 [0.968, 0.997] | 0.932 [0.903, 0.958]     |                           0.9775 |
| e5-base frozen + LR      | 0.967 [0.944, 0.986] | 0.928 [0.899, 0.954]     |                           0.9684 |
| TF-IDF + LR              | 0.964 [0.941, 0.984] | 0.932 [0.902, 0.958]     |                           0.9626 |
| CNN, ens5                | 0.971 [0.951, 0.987] | 0.932 [0.903, 0.958]     |                           0.9656 |
| qwen2.5-7b, 4-shot       | 0.922 [0.893, 0.951] | 0.942 [0.916, 0.968]     |                           0.9384 |
| qwen3-8b, 4-shot         | 0.925 [0.895, 0.951] | 0.931 [0.902, 0.958]     |                           0.9374 |

## Falsely labelled "delivery" groups

| model                    |   false_delivery_n |   predicted_other |   in_10pct_least_confident |   predicted_other_or_uncertain |   true_delivery_n |   true_delivery_predicted_other |   true_delivery_in_10pct_least_confident |
|:-------------------------|-------------------:|------------------:|---------------------------:|-------------------------------:|------------------:|--------------------------------:|-----------------------------------------:|
| e5-base fine-tuned, ens5 |                 22 |                 3 |                          8 |                              9 |               128 |                               2 |                                        6 |
| e5-base frozen + LR      |                 22 |                 5 |                         10 |                             11 |               128 |                               2 |                                        8 |
| TF-IDF + LR              |                 22 |                 6 |                         13 |                             14 |               128 |                               4 |                                       16 |
| CNN, ens5                |                 22 |                 5 |                          9 |                             11 |               128 |                               3 |                                        8 |
| qwen2.5-7b, 4-shot       |                 22 |                14 |                          7 |                             18 |               128 |                              10 |                                       17 |
| qwen3-8b, 4-shot         |                 22 |                12 |                          7 |                             16 |               128 |                               8 |                                       10 |
