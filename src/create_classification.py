from .generation import generate_task, AsyncList
import json
import re


CRITERION_PROMPT = '''
Ты - эксперт по оценке текстового содержимого сайтов и выполняешь задачу структурированной классификации текстового содержимого.

Твоя задача: по переданному фрагменту текста вернуть СТРОГО JSON с оценкой только по одному критерию.

Важно:
- Оценивай только на основе предоставленного фрагмента.
- Тебе дан один фрагмент текста, а не весь сайт.
- Не додумывай факты, которых нет во фрагменте.
- Не используй внешние знания о сайте, бренде, домене, стране, авторах или репутации ресурса.
- Не делай выводов о сайте в целом.
- Не пытайся быть "мягким" или "вежливым" в оценках - нужна максимально прагматичная классификация.
- Если данных недостаточно, обязательно отрази это в reasoning, написав: "Недостаточно данных для оценки".
- Не добавляй никаких пояснений вне JSON.
- Ответ должен быть валидным JSON.
- Сначала собери наблюдаемые признаки из текста, потом дай краткое обоснование, потом числовую оценку.
- score давай целым числом от 0 до 5.

Критерий: {CRITERION_NAME}

Описание:
{CRITERION_DESCRIPTION}

Шкала:
{CRITERION_SCALE}

Верни JSON строго в таком формате:

{{
  "reasoning": "string",
  "score": 0
}}

Требования к заполнению:
- reasoning: 1–2 коротких предложения, объясняющих, почему следует именно этот score.
- Никакого markdown, никаких комментариев, только JSON.

ФРАГМЕНТ ТЕКСТА:
{SITE_TEXT}
'''

CRITERIA = {
    "educational_value": {
        "description": "Насколько тексты полезны для обучения, расширения кругозора, получения системных знаний.",
        "scale": """0 - образовательной ценности почти нет
1 - очень низкая
2 - низкая
3 - средняя
4 - высокая
5 - очень высокая"""
    },
    "language_quality": {
        "description": """Оцени грамотность, связность, ясность, стиль, отсутствие грубых ошибок.
Учти, что тебе даются лишь небольшие вырезки разных текстов одного сайта, а не один связный текст.
Оценивай по видимым фрагментам: орфография, пунктуация, лексика - но не строй выводов о структуре всего текста""",
        "scale": """0 - очень плохое
1 - плохое
2 - ниже среднего
3 - среднее
4 - хорошее
5 - отличное"""
    },
    "commercial_intensity": {
        "description": "Насколько тексты похожи на рекламу, продажу, агрессивное продвижение услуг/товаров.",
        "scale": """0 - некоммерческий
1 - слабая коммерческая составляющая
2 - умеренная
3 - заметная
4 - высокая
5 - очень высокая"""
    },
    "rf_political_compliance": {
        "description": """Оцени, насколько тексты выглядят нейтральными / приемлемыми с точки зрения публичного информационного пространства РФ.
Это эвристическая оценка нейтральности и потенциальной спорности текста, а не юридический вывод.""",
        "scale": """0 - очень высокий риск неприемлемости
1 - высокий риск
2 - заметный риск
3 - умеренно нейтрально / спорно
4 - в целом нейтрально / приемлемо
5 - полностью нейтрально / безопасно"""
    },
    "information_reliability": {
        "description": """Оцени, насколько тексты выглядят фактически надежными по внутренним признакам:
- наличие конкретики,
- наличие аргументации,
- наличие оговорок и ограничений,
- отсутствие явных противоречий,
- отсутствие сенсационности без оснований,
- наличие ссылок на источники или проверяемых утверждений.""",
        "scale": """0 - крайне сомнительно
1 - низкая достоверность
2 - скорее низкая
3 - средняя / неясно
4 - высокая
5 - очень высокая"""
    },
    "harmful_content_risk": {
        "description": "Оцени наличие опасных советов, дезинформации, радикализации, дискриминации, экстремальности, призывов к вреду.",
        "scale": """0 - риска почти нет
1 - очень низкий
2 - низкий
3 - умеренный
4 - высокий
5 - очень высокий"""
    }
}

def clean_json(raw_text):
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw_text, re.DOTALL)
    if match:
        candidate = match.group(1).strip()
        return json.loads(candidate)
        
    return json.loads(raw_text.strip())


def create_prompt(site_texts_or_text: list, prompt_template: str, **kwargs):
    if isinstance(site_texts_or_text, str):
        site_text = site_texts_or_text[:4000]
    else:
        site_text = '\n\n-----------------\n\n'.join([f'{text[:4000]}...' for text in site_texts_or_text])
        
    prompt = prompt_template.format(SITE_TEXT=site_text, **kwargs)
    return prompt


async def classify(site: str, texts: list, out_f):
    tasks = AsyncList()
    task_meta = []

    criteria_order = list(CRITERIA.keys())
    for criterion_name in criteria_order:
        for text_idx, text in enumerate(texts):
            myprompt = create_prompt(
                text,
                CRITERION_PROMPT,
                CRITERION_NAME=criterion_name,
                CRITERION_DESCRIPTION=CRITERIA[criterion_name]["description"],
                CRITERION_SCALE=CRITERIA[criterion_name]["scale"]
            )
            tasks.append(generate_task(myprompt, max_tokens=250))
            task_meta.append((criterion_name, text_idx))

    await tasks.complete_couroutines(batch_size=36)
    tasks = await tasks.to_list()

    parsed_scores = {
        criterion_name: [] for criterion_name in criteria_order
    }

    parsed_reasonings = {
        criterion_name: [] for criterion_name in criteria_order
    }
    
    for meta, raw_result in zip(task_meta, tasks):
        criterion_name, text_idx = meta

        try:
            parsed = clean_json(raw_result)
            score = int(parsed["score"])
            reasoning = parsed["reasoning"]

            parsed_scores[criterion_name].append(score)
            parsed_reasonings[criterion_name].append({
                "text_idx": text_idx,
                "score": score,
                "reasoning": reasoning
            })
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue

    final_result = {
        "site": site,
        "criteria": {},
        "texts": texts,
    }

    for criterion_name in criteria_order:
        scores = parsed_scores[criterion_name]

        if not scores:
            return False

        if len(scores) < 5:
            return False

        score_mean = sum(scores) / len(scores)
        score_rounded = int(round(score_mean))

        final_result["criteria"][criterion_name] = {
            "score": score_rounded,
            "score_mean": score_mean,
            "per_text": parsed_reasonings[criterion_name]
        }

    out_f.write(json.dumps(final_result, ensure_ascii=False) + "\n")
    out_f.flush()
    return True