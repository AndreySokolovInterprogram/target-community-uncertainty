"""Manual check of the search-query labels: stratified random sample of 300 groups (150 + 150).

Output: annotation/manual_label_check_template.xlsx (sheet 'annotation' + sheet 'instructions'; the instruction text
shown to the annotator was in Russian, English version: annotation/class_definition.md) and a CSV copy of the sample.
The filled-in file is annotation/manual_label_check.xlsx (read by jise.label_check).
"""
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

from jise.common import ROOT, load_groups, save_json, run_info

SEED = 20261020
N_PER_CLASS = 150
OUT = ROOT / 'annotation'

INSTRUCTION = [
    'Проверка меток групп ВКонтакте (статья JISE, модуль 1)',
    '',
    'Что это. Метки групп получены не вручную, а по поисковому запросу VK, который нашёл группу:',
    '  «доставка» — группа найдена по запросам «доставка еды», «пицца», «суши», «бургеры», «роллы», «еда на дом», «доставка обедов»;',
    '  «прочее» — по запросам «новости», «юмор», «мемы», «фотография», «спорт», «книги».',
    'Нужно оценить, верна ли такая метка.',
    '',
    'Что считать «доставкой». Сообщество — служба доставки готовой еды: сама продаёт и доставляет (или отдаёт навынос)',
    'готовую еду — суши, пиццу, обеды, шашлык, выпечку и т.п. Сюда же относятся кафе и рестораны, если в группе есть',
    'доставка или заказ готовой еды, и сети/франшизы (например, «Додо Пицца» в конкретном городе).',
    'НЕ «доставка»: кулинарные блоги и рецепты, мукбанги, обзоры еды, продуктовые магазины без готовой еды, агрегаторы,',
    'вакансии курьеров, мемы и паблики «про еду», а также всё, что не связано с едой.',
    '',
    'Как размечать. Для каждой строки прочитайте название и описание; при необходимости откройте ссылку.',
    'В колонке «метка верна?» выберите:',
    '  да — метка в колонке «метка» соответствует определению выше;',
    '  нет — метка неверна (например, «доставка» стоит у блога с рецептами или «прочее» — у службы доставки);',
    '  непонятно — по названию, описанию и странице нельзя решить (группа удалена, закрыта, пустая и т.п.).',
    'В колонке «комментарий» можно коротко пояснить спорные случаи.',
    '',
    'Важно: оценивайте сообщество, а не то, как работает поиск. Не исправляйте метку, только ставьте да / нет / непонятно.',
    f'Выборка: случайная стратифицированная, {N_PER_CLASS} «доставка» + {N_PER_CLASS} «прочее», зерно {SEED}; строки перемешаны.',
]


def main():
    g = load_groups('title_desc')
    parts = [g[g['label'] == c].sample(N_PER_CLASS, random_state=SEED) for c in (1, 0)]
    s = pd.concat(parts).sample(frac=1, random_state=SEED).reset_index(drop=True)
    s.insert(0, 'n', range(1, len(s) + 1))
    s['link'] = 'https://vk.com/club' + s['group_id'].astype(str)
    s['label_name'] = s['label'].map({1: 'delivery', 0: 'other'})
    OUT.mkdir(parents=True, exist_ok=True)
    s[['n', 'group_id', 'link', 'title', 'description', 'label', 'label_name']].to_csv(
        OUT / 'manual_label_check_sample.csv', index=False)

    wb = Workbook()
    ws = wb.active
    ws.title = 'annotation'
    head = ['n', 'vk_id', 'link', 'title', 'description', 'label', 'label_correct', 'comment']
    ws.append(head)
    for r in s.itertuples():
        ws.append([r.n, int(r.group_id), r.link, r.title, r.description, r.label_name, None, None])
    for c in ws[1]:
        c.font = Font(bold=True)
        c.fill = PatternFill('solid', fgColor='DDDDDD')
        c.alignment = Alignment(wrap_text=True, vertical='top')
    widths = {'A': 5, 'B': 12, 'C': 30, 'D': 40, 'E': 80, 'F': 11, 'G': 18, 'H': 30}
    for col, w in widths.items():
        ws.column_dimensions[col].width = w
    for row in ws.iter_rows(min_row=2):
        for c in row:
            c.alignment = Alignment(wrap_text=True, vertical='top')
        row[2].hyperlink = row[2].value
        row[2].font = Font(color='0563C1', underline='single')
    dv = DataValidation(type='list', formula1='"yes,no,unclear"', allow_blank=True)
    ws.add_data_validation(dv)
    dv.add(f'G2:G{len(s) + 1}')
    ws.freeze_panes = 'A2'
    ws.auto_filter.ref = f'A1:H{len(s) + 1}'

    wi = wb.create_sheet('instructions')
    for line in INSTRUCTION:
        wi.append([line])
    wi['A1'].font = Font(bold=True, size=13)
    wi.column_dimensions['A'].width = 140
    wb.save(OUT / 'manual_label_check_template.xlsx')
    save_json(run_info({'seed': SEED, 'n_per_class': N_PER_CLASS, 'source_variant': 'title_desc (original text)',
                        'n_total': len(s), 'labels': s['label'].value_counts().to_dict()}),
              OUT / 'manual_label_check_info.json')
    print(len(s), s['label'].value_counts().to_dict(), 'empty descriptions:',
          int((s['description'].str.strip() == '').sum()))


if __name__ == '__main__':
    main()
