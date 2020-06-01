#!/usr/bin/env python
import os
import re
from collections import OrderedDict
from tika import parser
import textwrap
# parsed = parser.from_file("documents/LearningParseStructures.doc")
# parsed = parser.from_file("documents/LearningParseStructures.docx")
parsed = parser.from_file("D:\semantic\documents/LearningParseStructures.docx")

text = parsed["content"]
text = re.sub('\d+', 'NUMBER', text)
text = re.sub('NUMBER,NUMBER', '', text)
text = re.sub('NUMBER.NUMBER', '', text)
text = re.sub('NUMBER.', '', text)
text = re.sub('NUMBER', '', text)
text = re.sub(r'\t+', '', text)
text = re.sub(r'^$\n+', '', text, flags=re.MULTILINE)
text = re.sub("[!@#$+%*:()'-]", '', text)
text = os.linesep.join([s for s in text.splitlines() if s])
text = "\n".join(list(OrderedDict.fromkeys(text.split("\n"))))
text = re.sub('(?m)^.{0,50}\n','',text)

# paragraphs = text.split('   ')
paragraphs = text.split('\n')
for paragraph in paragraphs:
    print(paragraph)
    print('\n')
#print(text)
#print(paragraphs)
