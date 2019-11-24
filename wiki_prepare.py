# %%
import ujson
import pprint
from pathlib import Path
from tqdm import tqdm
from pythainlp.tokenize import word_tokenize
import re
# %%


def extract_txt_to_json(input_dir):
    input_paths = list(
        filter(lambda path: 'json' not in path.name, input_dir.glob('**/wiki_*')))
    # %%
    for input_path in tqdm(input_paths):
        with open(input_path, 'r', encoding='utf-8') as f:
            text = f.read()
        delimiter = '{"id":'
        json_texts = text.split(delimiter)
        json_texts = list(filter(lambda text: text != '', json_texts))
        json_texts = [delimiter+text for text in json_texts]
        try:
            json_texts = [ujson.loads(text) for text in json_texts]
            with open(input_path, 'w', encoding='utf-8') as f:
                ujson.dump(json_texts, f, ensure_ascii=False)
            input_path.rename(str(input_path)+'.json')
        except:
            pass


# %%
input_dir = Path('dataset/thwiki')
output_dir = Path('dataset/thwiki-words')

output_dir.mkdir(exist_ok=True)
with open('character_copus.txt', 'r', encoding='utf-8') as f:
    CHAR_CORPUS = f.read().replace('\n', '')
    CHAR_CORPUS = ''.join(sorted(list(set(CHAR_CORPUS))))
CHAR_DICT = {char: i for i, char in enumerate(CHAR_CORPUS)}
CHAR_DICT['$'] = len(CHAR_DICT)


input_paths = list(input_dir.glob('**/wiki_*.json'))
# %%
for input_path in tqdm(input_paths):
    with open(input_path, 'r', encoding='utf-8') as f:
        wiki_jsons = ujson.load(f)
    texts = [blog['text'] for blog in wiki_jsons]
    full_text = ''.join(texts)
    full_text = full_text.replace('\n', ' ').replace('  ', ' ')
    full_text = re.sub(f'[^{CHAR_CORPUS}]', '', full_text)
    full_text = full_text.replace(' ', '')

    words = word_tokenize(full_text, engine='mm')
    sub_parent_dir = output_dir / input_path.parent.name
    sub_parent_dir.mkdir(exist_ok=True)
    with open(sub_parent_dir / input_path.name, 'w', encoding='utf-8') as f:
        ujson.dump({'words': words}, f, ensure_ascii=False)


# %%
