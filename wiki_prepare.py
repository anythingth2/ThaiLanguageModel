# %%
import ujson
import pprint
from pathlib import Path
from tqdm import tqdm
# %%
input_dir = Path('dataset/thwiki')
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
