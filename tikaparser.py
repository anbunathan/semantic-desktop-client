#!/usr/bin/env python
import os
import re
from collections import OrderedDict
from tika import parser
from tika import config

class tikaparser:
    directory = ""
    filename = ""
    minimum_number_of_words = 5
    def __init__(sel):
        directory = ""
        filename = ""
        print("parser object created")

    def getConfig(self):
        print(config.getParsers())
        print(config.getMimeTypes())
        print(config.getDetectors())
        return config.getParsers()

    def parseDOC (self):
        filepath = os.path.join(self.directory, self.filename)
        parsed = parser.from_file(filepath)
        text = parsed["content"]
        chars_to_remove = ['.', '!', '?', '[', ']', '{', '}', '!', '@', '#', '$', '+', '%', '*', ':', '-', ',', '=',
                           '/', '\'']
        rx = '[' + re.escape(''.join(chars_to_remove)) + ']'
        text = re.sub('\d+', 'NUMBER', text)
        text = re.sub('NUMBER,NUMBER', '', text)
        text = re.sub('NUMBER.NUMBER', '', text)
        text = re.sub('NUMBER.', '', text)
        text = re.sub('NUMBER', '', text)
        text = re.sub(r'\t+', '', text)
        text = re.sub(r'^$\n+', '', text, flags=re.MULTILINE)
        text = re.sub("[!@#$+%*:()'-]", '', text)
        text = re.sub(rx, '', text)
        text = os.linesep.join([s for s in text.splitlines() if s])
        text = "\n".join(list(OrderedDict.fromkeys(text.split("\n"))))
        text = re.sub('(?m)^.{0,50}\n','',text)
        paragraphs = text.split('\n')
        paragraphs = [item for item in paragraphs if len(re.findall(r'\w+', item)) >
                      self.minimum_number_of_words]
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
        chars_to_remove = ['.', '!', '?', '[', ']', '{', '}', '!', '@', '#', '$', '+', '%', '*', ':', '-', ',', '=',
                           '/', '\'']
        rx = '[' + re.escape(''.join(chars_to_remove)) + ']'
        text = re.sub('\d+', 'NUMBER', text)
        text = re.sub('NUMBER,NUMBER', '', text)
        text = re.sub('NUMBER.NUMBER', '', text)
        text = re.sub('NUMBER.', '', text)
        text = re.sub('NUMBER', '', text)
        text = re.sub(r'\t+', '', text)
        text = re.sub(r'^$\n+', '', text, flags=re.MULTILINE)
        text = re.sub("[!@#$+%*:()'-]", '', text)
        text = re.sub(rx, '', text)
        text = os.linesep.join([s for s in text.splitlines() if s])
        text = "\n".join(list(OrderedDict.fromkeys(text.split("\n"))))
        text = re.sub('(?m)^.{0,50}\n', '', text)
        paragraphs = re.split(r'\s{3,}', text)
        paragraphs = [item for item in paragraphs if len(re.findall(r'\w+', item)) >
                      self.minimum_number_of_words]
        return paragraphs

    def parsePPT (self):
        filepath = os.path.join(self.directory, self.filename)
        parsed = parser.from_file(filepath)
        text = parsed["content"]
        chars_to_remove = ['.', '!', '?', '[', ']', '{', '}', '!', '@', '#', '$', '+', '%', '*', ':', '-', ',', '=',
                           '/', '\'']
        rx = '[' + re.escape(''.join(chars_to_remove)) + ']'

        text = re.sub('\d+', 'NUMBER', text)
        text = re.sub('NUMBER,NUMBER', '', text)
        text = re.sub('NUMBER.NUMBER', '', text)
        text = re.sub('NUMBER.', '', text)
        text = re.sub('NUMBER', '', text)
        text = re.sub(r'\t+', '', text)
        text = re.sub(r'^$\n+', '', text, flags=re.MULTILINE)
        text = re.sub("[!@#$+%*:()'-]", '', text)
        text = re.sub(rx, '', text)
        text = os.linesep.join([s for s in text.splitlines() if s])
        text = "\n".join(list(OrderedDict.fromkeys(text.split("\n"))))
        text = re.sub('(?m)^.{0,50}\n', '', text)
        paragraphs = text.split('\n')
        paragraphs = [item for item in paragraphs if len(re.findall(r'\w+', item)) >
                      self.minimum_number_of_words]
        return paragraphs

    def parseCSV (self):
        filepath = os.path.join(self.directory, self.filename)
        parsed = parser.from_file(filepath)
        text = parsed["content"]
        chars_to_remove = ['.', '!', '?', '[', ']', '{', '}', '!', '@', '#', '$', '+', '%', '*', ':', '-', ',', '=',
                           '/', '\'']
        rx = '[' + re.escape(''.join(chars_to_remove)) + ']'

        text = re.sub('\d+', 'NUMBER', text)
        text = re.sub('NUMBER,NUMBER', '', text)
        text = re.sub('NUMBER.NUMBER', '', text)
        text = re.sub('NUMBER.', '', text)
        text = re.sub('NUMBER', '', text)
        text = re.sub(r'\t+', '', text)
        text = re.sub(r'^$\n+', '', text, flags=re.MULTILINE)
        text = re.sub("[!@#$+%*:()'-]", '', text)
        text = re.sub(rx, '', text)
        text = os.linesep.join([s for s in text.splitlines() if s])
        text = "\n".join(list(OrderedDict.fromkeys(text.split("\n"))))
        text = re.sub('(?m)^.{0,50}\n', '', text)
        paragraphs = text.split('\n')
        paragraphs = [item for item in paragraphs if len(re.findall(r'\w+', item)) >
                      self.minimum_number_of_words]
        return paragraphs

    def parseXLS (self):
        filepath = os.path.join(self.directory, self.filename)
        parsed = parser.from_file(filepath)
        text = parsed["content"]
        chars_to_remove = ['.', '!', '?', '[', ']', '{', '}', '!', '@', '#', '$', '+', '%', '*', ':', '-', ',', '=',
                           '/', '\'']
        rx = '[' + re.escape(''.join(chars_to_remove)) + ']'

        text = re.sub('\d+', 'NUMBER', text)
        text = re.sub('NUMBER,NUMBER', '', text)
        text = re.sub('NUMBER.NUMBER', '', text)
        text = re.sub('NUMBER.', '', text)
        text = re.sub('NUMBER', '', text)
        text = re.sub(r'\t+', '', text)
        text = re.sub(r'^$\n+', '', text, flags=re.MULTILINE)
        text = re.sub("[!@#$+%*:()'-]", '', text)
        text = re.sub(rx, '', text)
        text = os.linesep.join([s for s in text.splitlines() if s])
        text = "\n".join(list(OrderedDict.fromkeys(text.split("\n"))))
        text = re.sub('(?m)^.{0,20}\n', '', text)
        paragraphs = text.split('\n')
        paragraphs = [item for item in paragraphs if len(re.findall(r'\w+', item)) >
                      self.minimum_number_of_words]
        return paragraphs

    def parse (self, directory, filename):
        self.directory = directory
        self.filename = filename
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
