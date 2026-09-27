"""Explainable pairing heuristics, not a probability or a trained taste model."""
import re

# Columns: white, red, rose, orange, sparkling. Scores are product heuristics.
FOODS = {
    'meat': ('мясо', r'мяс\w*|говя\w*|стейк\w*|баран\w*|свинин\w*|шашлык\w*|бургер\w*|бефстроганов\w*', [40, 85, 65, 60, 55], 'Для насыщенного мяса обычно стоит попробовать более плотное красное вино.'),
    'poultry': ('птица', r'куриц\w*|курин\w*|цыплен\w*|индей\w*|птиц\w*|утк\w*', [80, 65, 80, 70, 75], 'Для птицы попробуйте белое или розовое; к насыщенному соусу можно взять лёгкое красное.'),
    'fish': ('рыба', r'рыб\w*|лосос\w*|семг\w*|форел\w*|треск\w*|судак\w*|тун\w*|дорад\w*|сибас\w*', [85, 40, 75, 65, 80], 'К нежной рыбе попробуйте белое вино, а к жирной рыбе — также розовое.'),
    'seafood': ('морепродукты', r'морепродукт\w*|кревет\w*|миди\w*|устриц\w*|кальмар\w*|краб\w*|гребеш\w*|суши|ролл\w*', [90, 30, 75, 55, 90], 'Для морепродуктов попробуйте свежее белое или сухое игристое.'),
    'vegetables': ('овощи', r'овощ\w*|салат\w*|кабач\w*|баклаж\w*|брокколи|спарж\w*', [80, 45, 80, 70, 75], 'К овощам подойдут свежие белые или розовые стили; важны заправка и способ приготовления.'),
    'mushrooms': ('грибы', r'гриб\w*|шампиньон\w*|трюфел\w*', [65, 75, 60, 75, 55], 'К грибам попробуйте лёгкое красное или более насыщенное белое вино.'),
    'cheese': ('сыр', r'сыр\w*|пармезан\w*|бри|камамбер\w*|моцарелл\w*|горгонзол\w*', [75, 70, 70, 75, 85], 'Уточните сыр: мягкий, выдержанный и с плесенью требуют разных сочетаний.'),
    'dessert': ('десерт', r'десерт\w*|торт\w*|пирожн\w*|шоколад\w*|морожен\w*|тирамису|чизкейк\w*', [25, 30, 30, 30, 35], 'К сладкому десерту попробуйте вино не менее сладкое, чем само блюдо.'),
    'pasta': ('паста или пицца', r'паст\w*|макарон\w*|пицц\w*|ризотто|карбонара|болоньезе', [65, 65, 70, 60, 65], 'Уточните соус: сливочный, томатный или мясной — он существенно влияет на сочетание.'),
}
STYLES = ['white', 'red', 'rose', 'orange', 'sparkling']
STYLE_NAMES = ['белое', 'красное', 'розовое', 'оранжевое', 'игристое']
DISCLAIMER = 'Условная оценка по правилам, а не вероятность и не результат дегустации. Она относится к показанной карточке вина; если вино распознано неверно, рекомендация тоже может не подойти.'


def normalize(text):
    return str(text or '').lower().replace('ё', 'е')


def matches(text, pattern):
    return bool(re.search(r'\b(?:' + pattern + r')\b', text))


def assess_pairing(card, dish):
    text = normalize(dish)
    # Simple exclusions: don't interpret "без мяса" or "не острый" as present.
    text = re.sub(r'\b(?:без|не)\s+[а-яa-z]+', '', text)
    groups = [key for key, (_, pattern, _, _) in FOODS.items() if matches(text, pattern)]
    identity = normalize(' '.join(str(card.get(k, '') or '') for k in
                                 ['Категория', 'Название вина', 'Slug', 'Сахар', 'Тип вина']))
    category = normalize(card.get('Категория'))
    if matches(identity, r'игрист\w*|брют\w*|брют|спуманте|spumante|bryut|brut|igrist\w*'):
        style = 'sparkling'
    else:
        style = next((s for s, prefix in [('rose', 'роз'), ('orange', 'оранж'), ('red', 'крас'), ('white', 'бел')]
                      if prefix in category), None)
    sweetness = ('semi' if matches(identity, r'полуслад\w*|poluslad\w*') else
                 'sweet' if matches(identity, r'сладк\w*|sladk\w*|десерт\w*') else
                 'dry' if matches(identity, r'сух\w*|suho\w*|брют\w*|brut|bryut') else 'unknown')
    result = {'dish': dish, 'score': None, 'score_max': 100, 'method': 'rules-v1',
              'disclaimer': DISCLAIMER, 'recognized_foods': [FOODS[g][0] for g in groups],
              'wine_style': STYLE_NAMES[STYLES.index(style)] if style else None,
              'reasons': [], 'recommendations': []}
    if not groups or not style:
        result.update(status='needs_details', label='Нужно уточнение', explanation=(
            'Не удалось определить блюдо. Укажите основной продукт и соус, например: «лосось в сливочном соусе» или «стейк на гриле».'
            if not groups else 'В карточке недостаточно данных о стиле вина. Оценку совместимости не рассчитываем.'))
        return result
    # A sauce/starch side should not outweigh an explicitly named main ingredient.
    if len(groups) > 1 and 'pasta' in groups:
        groups.remove('pasta')
    column = STYLES.index(style)
    base = round(sum(FOODS[g][2][column] for g in groups) / len(groups))
    reasons = [f'Основа оценки: {result["wine_style"]} вино и {", ".join(FOODS[g][0] for g in groups)}.']
    tips = [FOODS[g][3] for g in groups]
    if matches(text, r'остр\w*|чили|перец|карри|аджик\w*'):
        delta = 5 if sweetness in ('semi', 'sweet') else -20 if style == 'red' else -10
        base += delta
        reasons.append('Учтена острота: небольшая сладость может смягчить жжение; алкоголь и танины могут его усилить.')
        tips.append('Уменьшите остроту соуса или попробуйте полусухое/полусладкое вино с невысоким алкоголем.')
    if matches(text, r'грил\w*|жарен\w*|копчен\w*|барбекю'):
        base += 5 if style in ('red', 'orange', 'sparkling') else -5
        reasons.append('Учтены жарка, гриль или копчение: вкус блюда становится интенсивнее.')
    if matches(text, r'сливоч\w*|сливк\w*'):
        base += 5 if style in ('white', 'sparkling') else -5
        reasons.append('Учтён сливочный соус: свежесть белого или игристого может уравновесить насыщенность блюда.')
    if matches(text, r'лимон\w*|томат\w*|уксус\w*'):
        base += 5 if style in ('white', 'rose', 'sparkling') else -5
        reasons.append('Учтена кислая заправка: стоит проверить, достаточно ли свежести у вина.')
    if 'dessert' in groups:
        if sweetness == 'sweet':
            base = 80
            reasons.append('В карточке указан сладкий стиль: он перспективнее для десерта, чем сухой.')
        elif sweetness == 'semi':
            base = 60
            reasons.append('Вино полусладкое: очень сладкий десерт может оказаться слаще вина.')
        elif sweetness == 'unknown':
            result.update(status='needs_details', label='Нужно уточнение',
                          explanation='Для десерта нужно знать сладость вина. В карточке она не указана явно; оценку не рассчитываем.',
                          recommendations=tips)
            return result
        else:
            reasons.append('Сладость десерта может сделать сухое вино более резким и горьким.')
    elif sweetness in ('sweet', 'semi'):
        base -= 10 if sweetness == 'sweet' else 5
        reasons.append('Учтена сладость вина: сочетание с несладким блюдом требует осторожности.')
    if sweetness == 'unknown':
        reasons.append('Сладость вина не указана явно, поэтому в оценке она не учитывается.')
    score = max(10, min(95, 5 * round(base / 5)))
    label = 'Перспективное сочетание' if score >= 75 else 'Зависит от приготовления' if score >= 50 else 'Лучше изменить сочетание'
    result.update(score=score, status='estimated', label=label, reasons=reasons,
                  recommendations=tips, explanation=f'Для блюда «{dish}»: {label.lower()}. ' + ' '.join(reasons))
    return result
