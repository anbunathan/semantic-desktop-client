#!/usr/bin/env python
import os
import re
from collections import OrderedDict
from tika import parser
import textwrap

class tikaparser:
    def __init__(self, directory, filename):
        self.directory = directory
        self.filename = filename

    def parseDOC (self):
        filepath = os.path.join(self.directory, self.filename)
        parsed = parser.from_file(filepath)
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
        paragraphs = text.split('\n')
        return paragraphs

    def parsePDF (self):
        filepath = os.path.join(self.directory, self.filename)
        raw = parser.from_file(filepath)
        raw = raw["content"]
        raw = str(raw)
        safe_text = raw.encode('utf-8', errors='ignore')
        safe_text = str(safe_text).replace("\n", "").replace("\\", "$$$$")
        safe_text = safe_text.replace('$$$$n', '').replace("b\'", '')
        text = safe_text
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
        text = re.sub('(?m)^.{0,50}\n', '', text)
        paragraphs = re.split(r'\s{3,}', text)
        return paragraphs

    def parsePPT (self):
        filepath = os.path.join(self.directory, self.filename)
        parsed = parser.from_file(filepath)
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
        text = re.sub('(?m)^.{0,50}\n', '', text)
        paragraphs = text.split('\n')
        return paragraphs

    def parseCSV (self):
        filepath = os.path.join(self.directory, self.filename)
        parsed = parser.from_file(filepath)
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
        text = re.sub('(?m)^.{0,50}\n', '', text)
        paragraphs = text.split('\n')
        return paragraphs

    def parseXLS (self):
        filepath = os.path.join(self.directory, self.filename)
        parsed = parser.from_file(filepath)
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
        text = re.sub('(?m)^.{0,20}\n', '', text)
        paragraphs = text.split('\n')
        return paragraphs

    def parse (self):
        paragraphs = None
        filepath = os.path.join(self.directory, self.filename)
        filename, file_extension = os.path.splitext(filepath)
        print(file_extension)
        if (file_extension == '.doc' or file_extension == '.docx' or file_extension == '.txt'):
            paragraphs = self.parseDOC()
        elif (file_extension == '.pdf'):
            paragraphs = self.parsePDF()
        elif (file_extension == '.ppt' or file_extension == '.pptx'):
            paragraphs = self.parsePPT()
        elif (file_extension == '.csv'):
            paragraphs = self.parseCSV()
        elif (file_extension == '.xls' or file_extension == '.xlsx'):
            paragraphs = self.parseXLS()
        return paragraphs