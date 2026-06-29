import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from vllm import LLM
from tqdm import tqdm


DATA_DIR = Path('mini_fineweb2_rus_all_data')
INDEX_FILE = Path('legal_domain.jsonl')
OUTPUT_FILE = Path('legal_domain_data_fixed.jsonl')
MODEL_PATH = 'bert_classifier/model_separate_text_train/educational_value/checkpoint-7500'

llm = LLM(
    model=str(MODEL_PATH),
    max_model_len=8192,
)
tokenizer = llm.get_tokenizer()

entries = []
with open(INDEX_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        entries.append(json.loads(line.strip()))

file_to_entries: dict[list[str]] = defaultdict(list)
for ent in entries:
    file_to_entries[ent['file_name']].append(ent)


results: dict[str, int] = {}

with open(OUTPUT_FILE, 'w', encoding='utf-8') as fout:
    for file_name, file_entries in tqdm(file_to_entries.items(), desc="Обработка файлов"):
        file_path = DATA_DIR / file_name
    
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [json.loads(line.strip())['text'] for line in f]
    
        batch_texts = []
        idx_to_ent = {}
        for ent in file_entries:
            idx = ent['idx']
            batch_texts.append(lines[idx])
            idx_to_ent[idx] = ent
    
        encodings = tokenizer(
            batch_texts,
            truncation=True,
            max_length=llm.llm_engine.model_config.max_model_len,
            add_special_tokens=True,
        )
    
        outputs = llm.classify(encodings["input_ids"])
        logits = [output.outputs.probs for output in outputs]
        logits = np.array(logits, dtype=np.float32).reshape(-1)
    
        probs = 1.0 / (1.0 + np.exp(-logits))
        scores_cont = probs * 5.0
        scores_disc = np.clip(np.rint(scores_cont), 0, 5).astype(int)
    
        for text_idx, prob, sd in zip(
            [ent['idx'] for ent in file_entries if ent['idx'] in idx_to_ent],
            probs,
            scores_disc
        ):
            orig_ent = idx_to_ent[text_idx]
            new_ent = {
                **orig_ent,
                'educational_value_prob': float(prob),
                'educational_value_discrete': int(sd)
            }
            fout.write(json.dumps(new_ent, ensure_ascii=False) + '\n')

print(f"Результат записан в {OUTPUT_FILE}")