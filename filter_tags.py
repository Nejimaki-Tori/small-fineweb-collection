import asyncio
import json
from base_llm_request.generation import generate_task, AsyncList


LEGAL_TAGS = '''
Ты - ассистент для классификации тегов по принадлежности к юридическому домену.

Юридический домен включает всё, что прямо связано с:
- правом, законодательством, нормативными актами;
- судебной системой, адвокатурой, нотариатом, прокуратурой;
- юридическими профессиями, правовыми статусами;
- договорами, сделками, правами и обязанностями;
- преступлениями, правонарушениями, наказаниями, уголовным/гражданским процессом;
- иными специализированными правовыми понятиями.

Тег НЕ относится к юридическому домену, если он:
- описывает другую профессию, хобби, сферу деятельности (спорт, кулинария, медицина, журналистика);
- является общеупотребительным словом без специфического правового смысла.

Ты получаешь один тег и возвращаешь JSON-объект:
reasoning - краткое обоснование твоего ответа на русском языке,
YES - если тег относится к юридическому домену,
NO - если не относится.

Никаких пояснений, комментариев и дополнительного текста не добавляй.

Примеры:
Тег: #право
YES
Тег: #журналистика
NO
Тег: #закон
YES
Тег: #спорт
NO
Тег: #суды
YES
Тег: #рецепт
NO
Тег: #налогообложение
YES
Тег: #погода
NO

Ответь **только** JSON-объектом, без каких-либо дополнительных символов или текста вне JSON.  
Формат строго такой:
{{"reasoning": "краткое обоснование твоего ответа на русском языке", "answer": "YES" или "NO"}}


Теперь обработай следующий тег:
Тег: {tag}
'''


def parse_classification_json(raw_text: str):
    try:
        data = json.loads(raw_text.strip())
        reasoning = data.get("reasoning", "")
        answer = data.get("answer", "").strip().upper()
        if answer not in ("YES", "NO"):
            answer = "NO"  # fallback
        return reasoning, answer
    except json.JSONDecodeError:
        return "", "NO"

def build_prompt(tag: str) -> str:
    return LEGAL_TAGS.format(tag=tag)

async def classify_tags(tags: list[str], batch_size=40, verbose=True):
    results = AsyncList()
    for tag in tags:
        prompt = build_prompt(tag)
        results.append(generate_task(prompt, post_process=False))

    await results.complete_couroutines(batch_size=batch_size, verbose=verbose)

    decisions = {}
    reasoning_dict = {}
    for tag, raw in zip(tags, results):
        reasoning, answer = parse_classification_json(raw)
        reasoning_dict[tag] = reasoning
        decisions[tag] = answer == "YES"
    return decisions, reasoning_dict

with open('bert_tagging/model/multilabel_llm_no_abbriv_tags/checkpoint-18500/config.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    id2label = data['id2label']
    tags = list(id2label.values())

answers, reasoning_dict = asyncio.run(classify_tags(tags))

with open('tag_map_legal_reasoning.json', 'w', encoding='utf-8') as f:
    json.dump(answers, f, ensure_ascii=False, indent=4)