import gzip
import json
from tqdm import tqdm
load_fts = ['reviewerID', 'asin', 'overall', 'category', 'brand', 'title', 'description']
ft2idx = {k : {} for k in load_fts}
iid2ft = {}
card = {k : 1 for k in load_fts}

def fliter(text):
  if 'href' in text or '<' in text or 'http' in text or '=' in text or '_' in text or '?' in text or '>' in text or '\n' in text:
    return False
  else:
    return True

def get_internal_id(field, ft):
  if '_seq' in field:
    ans = []
    for ks in ft:
      ans.append(get_internal_id(field.replace('_seq',''),  ks))
    return ans
  ft = ft.replace('\\n','')
  ft = ft.replace('\n','')
  if ft not in ft2idx[field]:
    ft2idx[field][ft] = card[field]
    card[field] += 1
  return ft2idx[field][ft]

def parse_feature(file_name):
  if isinstance(file_name, list):
    for file in  file_name:
      parse_feature(file)
    return
  
  g = gzip.open(file_name, 'r')
  for l in tqdm(g):
      l = json.dumps(eval(l))
      record = json.loads(l)
      try:
        iid, brand, title, desc = record['asin'], record['brand'], record['title'], record['description']
      except:
        continue
      cat =  record['category'] if 'category' in record else record['categories'][0]
      if not isinstance(desc, list):
        desc = [desc]
      desc = list(filter(fliter, desc))
      if '\n' in brand:
        continue
      if len(desc) == 0 or len(cat) == 0 or len(title) == 0:
        continue
      else:
        desc = ' '.join(desc)
      iid, brand, title, desc, cat = get_internal_id('asin',iid),\
         get_internal_id('brand',brand),\
            get_internal_id('title_seq',title.split(' ')),\
               get_internal_id('description_seq',desc.split(' ')),\
                  get_internal_id('category_seq',cat)
      iid2ft[iid] = [iid, brand, title, desc, cat]

def parse_inter(file_name):
    if isinstance(file_name, list):
      for file in  file_name:
        parse_inter(file)
      return

    all_data = []
    g = gzip.open(file_name, 'r')
    for l in tqdm(g):
        record = json.loads(l)
        if int(record['overall']) == 3:
          continue
        uid, iid, label = record['reviewerID'], record['asin'], int(int(record['overall']) >= 4)
        if iid not in ft2idx['asin']:
          continue
        uid, iid = get_internal_id('reviewerID', uid), get_internal_id('asin', iid)
        ift = iid2ft[iid]
        all_data.append([label, uid] + ift)

    with open(file_name + '.csv', 'w+') as f:
      for line in tqdm(all_data):
        pline = []
        for idx, ss in enumerate(line):
          if idx < len(line) - 3:
            pline.append(str(ss))
          else:
            pline.append(' '.join(map(str, ss)))
        f.write(','.join(pline) + '\n')

dataset = ['Electronics','Toys_and_Games', 'Office_Products', 'Musical_Instruments', 'Grocery_and_Gourmet_Food', 'Books', 'Movies_and_TV']

def Single_to_double_quote(path):
    fileObject = gzip.open(path, 'r', encoding='utf-8').read()
    dels = fileObject.replace('\'', '\"')
    open(path, 'w', encoding='utf-8').write(dels)

parse_feature(['meta_' + x + '.json.gz' for x in dataset])
parse_inter(['reviews_' + x + '_5.json.gz' for x in dataset])

feature_index = open('./feature_index', 'w')
for fname, field in ft2idx.items():
    for feat, id in field.items():
        assert '\n' not in feat and '\\n' not in feat
        feature_index.write('%s|raw_feat_%s|%s\n' % (fname, str(feat), id))
feature_index.close()
print(card)