Food queries (label 1): доставка еды, пицца, суши, бургеры, роллы, еда на дом, доставка обедов; non-food (label 0): новости, юмор, мемы, фотография, спорт, книги (later items of the lists were never reached).

| field       |   label1_with_food_query_word |   label0_with_nonfood_query_word |   label1_with_nonfood_word_only |   label0_with_food_word_only |   rule_accuracy |
|:------------|------------------------------:|---------------------------------:|--------------------------------:|-----------------------------:|----------------:|
| title       |                        0.9908 |                           0.9951 |                          0.0002 |                       0.0000 |          0.9952 |
| description |                        0.8826 |                           0.7382 |                          0.0058 |                       0.0020 |          0.9321 |
| text        |                        0.9982 |                           0.9983 |                          0.0009 |                       0.0000 |          0.9883 |

Masked variants: where the errors are

| text              | model     |     n |   n_empty_after_mask |   acc_all |   acc_nonempty |   acc_empty |   share_of_errors_on_empty |
|:------------------|:----------|------:|---------------------:|----------:|---------------:|------------:|---------------------------:|
| title_desc_masked | e5base_lr | 11612 |                  501 |    0.9684 |         0.9692 |      0.9501 |                     0.0681 |
| title_desc_masked | tfidf_lr  | 11612 |                  501 |    0.9626 |         0.9633 |      0.9481 |                     0.0599 |
| title_desc_masked | cnn ens5  | 11612 |                  501 |    0.9656 |         0.9665 |      0.9461 |                     0.0677 |
| desc_masked       | e5base_lr | 10606 |                  123 |    0.9702 |         0.9704 |      0.9512 |                     0.0190 |
| desc_masked       | tfidf_lr  | 10606 |                  123 |    0.9661 |         0.9663 |      0.9431 |                     0.0194 |

Possible label noise among label 1: {'label1': 5656, 'label1_without_delivery_or_order_words': 712, 'label1_recipe_like_without_order_words': 143, 'note': 'heuristic upper/lower bounds, not manually verified'}
