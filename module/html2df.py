import pandas as pd
import bs4
from bs4 import BeautifulSoup
import re
import chardet


class HTML2df():

    def __init__(self):
        print("Create HTML2df Object.")
        self.tagIgnore = {'head',
                          'iframe',
                          'script',
                          'meta',
                          'link',
                          'style',
                          'input',
                          'checkbox',
                          'button',
                          'noscript'}

    def rmDuplicateSpace(self, sentence):
        result = ""
        space_count = -1
        for c in sentence:
            if c == " ":
                space_count += 1
                if space_count == 0 and result != "":
                    result += c
                    continue
            else:
                space_count = -1
                result += c
        if len(result) > 0 and result[-1] == " ":
            result = result[:-1]
        return result

    def _traversalTree(self, root, depth, prefix="", label=0):
        if type(root) is bs4.element.NavigableString:
            thisName = " " if root.name is None else root.name+" "
            content = self.rmDuplicateSpace(
                str(root).replace(
                    "\n",
                    " "
                ).replace(
                    "\xa0",
                    " "
                ).replace(
                    "\u3000",
                    " "
                )
            )
            return [content], [prefix+thisName], [label]

        if type(root) is bs4.element.Tag and root.name not in self.tagIgnore:
            if root.has_attr('__boilernet_label'):
                label = int(root['__boilernet_label'])
            result = []
            tag = []
            label_list = []
            for child in root.contents:
                c_result, c_tag_list, c_label_list = self._traversalTree(
                    child,
                    depth,
                    prefix=prefix+root.name+" ",
                    label=label)
                result += c_result
                tag += c_tag_list
                label_list += c_label_list
            if len(prefix.split(" ")) == depth:
                concat_result = ""
                for r in result:
                    concat_result += " " + r
                concat_tag = prefix + root.name
                concat_label = [1] if 1 in label_list else [0]
                result = [concat_result]
                tag = [concat_tag]
                label_list = concat_label
            return result, tag, label_list
        return [], [], []

    def convert2df(self,
                   htmlstr,
                   generate_label=False,
                   depth=float("inf")):
        soup = BeautifulSoup(htmlstr, 'lxml')
        leafnodes, tags, labels = self._traversalTree(soup.html, depth)
        if generate_label:
            df = pd.DataFrame(
                {"tag": tags, "content": leafnodes, "label": labels})
        else:
            df = pd.DataFrame({"tag": tags, "content": leafnodes})
        df['depth'] = [len(list(filter(None, t.split(" ")))) for t in df.tag]
        dropList = []
        for i in df.index:
            if df['content'][i] == "":
                dropList.append(i)
        df = df.drop(dropList, axis='index',
                     inplace=False).reset_index(drop=True)
        return df

    def file2df(self,
                filepath,
                encoding=None,
                generate_label=False,
                depth=float("inf")):
        if encoding is None:
            with open(filepath, 'rb') as file:
                d = file.read()
            encoding = chardet.detect(d)['encoding']
            encoding = encoding if encoding != 'GB2312' else 'GB18030'
        with open(filepath, 'r', encoding=encoding, errors='ignore') as file:
            html = file.read()
        html = html.replace("&lt;", "<").replace("&gt;", ">")
        return self.convert2df(html, generate_label, depth)
