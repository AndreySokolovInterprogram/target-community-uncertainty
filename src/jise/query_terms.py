"""Search queries that produced the labels (notebook 'Get data.ipynb', groups.search) and a masking function.

Label 1 = group returned by a food-delivery query, label 0 = returned by a non-food query.
Masking removes every token that starts with a query stem (Russian forms + common English/translit forms),
so a model cannot simply recognise the query word.
"""
import re

POS_QUERIES = ['доставка еды', 'пицца', 'суши', 'бургеры', 'роллы', 'еда на дом', 'доставка обедов',
               'доставка шашлыка', 'доставка пирогов', 'кулинария', 'catering']   # last 4 listed but not reached
NEG_QUERIES = ['новости', 'юмор', 'мемы', 'фотография', 'спорт', 'книги', 'фильмы', 'музыка',
               'путешествия', 'искусство']                                      # last 4 listed but not reached
POS_STEMS = ['достав', 'пицц', 'суш', 'бургер', 'ролл', 'обед', 'шашлык', 'пирог', 'кулинар', 'кейтер',
             'pizz', 'sushi', 'burger', 'roll', 'deliver', 'cater', 'food']
POS_WORDS = ['еда', 'еды', 'еду', 'едой', 'еде', 'дом', 'дома', 'на']   # 'еда на дом' (short words: exact match)
NEG_STEMS = ['новост', 'юмор', 'мем', 'фотограф', 'фото', 'спорт', 'книг', 'книж', 'фильм', 'музык',
             'путешеств', 'искусств', 'news', 'humor', 'humour', 'meme', 'photo', 'sport', 'book', 'film',
             'movie', 'music', 'travel']
NEG_WORDS = ['art']
TOKEN = re.compile(r'\w+', re.UNICODE)
_stem_re = re.compile(r'^(?:' + '|'.join(map(re.escape, POS_STEMS + NEG_STEMS)) + ')', re.IGNORECASE)
_words = set(POS_WORDS + NEG_WORDS)


def is_query_token(tok):
    t = tok.lower()
    return bool(_stem_re.match(t)) or t in _words


def mask_text(s):
    """Replace query tokens by nothing (keeps the rest of the text, incl. punctuation)."""
    return TOKEN.sub(lambda m: '' if is_query_token(m.group(0)) else m.group(0), s)


def has_terms(s, stems, words=()):
    rs = re.compile(r'^(?:' + '|'.join(map(re.escape, stems)) + ')', re.IGNORECASE)
    return any(rs.match(t) or t.lower() in words for t in TOKEN.findall(s))
