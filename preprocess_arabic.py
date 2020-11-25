from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re,sys,codecs
import json

def clean_text(text):
    # remove tashkeel
    text = text.replace('{', 'ا')
    text = text.replace('}', 'ا')

    #text = text.replace('-','')
    p_tashkeel = re.compile(r'[\u0617-\u061A\u064B-\u0652]')
    text = re.sub(p_tashkeel, "", text)

    #Other typos in the conll files
    text = text.replace('ه`ذا', 'هذا')
    text = text.replace('ه`ذه', 'هذه')
    text = text.replace('ه`ذين', 'هذين')
    text = text.replace('الل`ه','الله')
    text = text.replace('ذ`لك', 'ذلك')
    text = text.replace('إل`ه','إله')
    return text

if __name__ == '__main__':
    if len(sys.argv) < 3:
        sys.exit("Usage: {} <input_path> <output_path>.".format(sys.argv[0]))
    writer = codecs.open(sys.argv[2],'wb',encoding='utf-8')
    for line in codecs.open(sys.argv[1],encoding='utf-8'):
        doc = json.loads(line)
        sentences = doc['sentences']
        clean_sentences = [[clean_text(t) for t in sent] for sent in sentences]
        doc['sentences'] = clean_sentences
        writer.write(json.dumps(doc)+'\n')
