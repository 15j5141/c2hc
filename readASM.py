import os
import json
import re

current_dir = os.path.dirname(os.path.abspath(__file__)) + "\\"
print("abs dirname: ", current_dir)

class ReadASM:
    def __init__(self):
        self.path_mnemonic = "memo\\mnemonic.json"
        self.path_section = "memo\\section.json"

    def load(self, filename=None):
        self.lines = []
        current_dir = os.path.dirname(os.path.abspath(__file__)) + "\\"
        if filename is None:
            path = current_dir + "memo\\main.s"
        else:
            path = filename
        with open(self.path_mnemonic, "r") as pt:
            json_mnemonic = json.load(pt)
        with open(self.path_section, "r") as pt:
            json_section = json.load(pt)
        self.mnemonic_section_list = json_mnemonic + json_section
        with open(path) as f:
            for s_line in f:
                self.lines.append(s_line)
    def parse(self):
        matcher_label = re.compile("[a-zA-Z0-9.]+:")
        matcher_nolabel = re.compile("\s+[a-zA-Z0-9.]+")
        matcher_mnemonic_func = lambda x: re.compile("\s+" + str.lower(x) + "l?q?\W")
        results = []

        # 各行ごとに処理を行う.
        for s_line in self.lines:
            # print(s_line)
            if matcher_label.match(s_line):
                # ラベルを固定値で結果一覧に格納する.
                print("label" + s_line)
                results.append(str.upper("__LABEL__"))
            elif matcher_nolabel.match(s_line):
                # filter
                # checked = [x for x in json_mnemonic if matcher_mnemonic_func(x).match(s_line)]
                # checked2 = [x for x in json_section if matcher_mnemonic_func(x).match(s_line)]
                checked0 = [
                    x for x in self.mnemonic_section_list if matcher_mnemonic_func(x).match(s_line)
                ]
                # 結果一覧に格納する.
                if len(checked0) > 0:
                    print("mnemonic or section" + str(checked0))
                    results.append(str.upper(checked0[0]))
                    continue
                else:
                    print("other" + str(s_line))
                    continue
                if len(checked) > 0:
                    print("mnemonic" + str((checked)))
                    results.append(str.upper(checked[0]))
                elif len(checked2) > 0:
                    print("section" + str((checked2)))
                else:
                    print("other[" + s_line)
        return results

    def save(self, results):
        with open("memo\\results.json", "w") as f:
            json.dump(results, f, indent=4)

reader = ReadASM()
reader.load()
result = reader.parse()
print(result)