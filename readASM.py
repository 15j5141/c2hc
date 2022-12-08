import os
import json
import re

current_dir = os.path.dirname(os.path.abspath(__file__)) + "\\"
print("abs dirname: ", current_dir)


with open("memo\\mnemonic.json", "r") as pt:
    json_mnemonic = json.load(pt)

with open("memo\\section.json", "r") as pt:
    json_section = json.load(pt)

print(type(json_mnemonic))
print((json_section))

mnemonic_section_list = json_mnemonic + json_section

results = []
path = current_dir + "memo\\main.s"
lines = []
matcher_label = re.compile("[a-zA-Z0-9.]+:")
matcher_nolabel = re.compile("\s+[a-zA-Z0-9.]+")
matcher_mnemonic_func = lambda x: re.compile("\s+" + str.lower(x) + "l?q?\W")

# ファイルから各行ごとに取り出す.
with open(path) as f:
    for s_line in f:
        lines.append(s_line)

# 各行ごとに処理を行う.
for s_line in lines:
    # print(s_line)
    if matcher_label.match(s_line):
        print("label"+s_line)
        results.append(str.upper("__LABEL__"))
    elif matcher_nolabel.match(s_line):
        checked = [x for x in json_mnemonic if matcher_mnemonic_func(x).match(s_line)]
        checked2 = [x for x in json_section if matcher_mnemonic_func(x).match(s_line)]
        checked0 = [x for x in mnemonic_section_list if matcher_mnemonic_func(x).match(s_line)]
        if len(checked0) > 0:
            print("mnemonic or section"+ str(checked0))
            results.append(str.upper(checked0[0]))
            continue
        else:
            print("other"+ str(s_line))
            continue
        if len(checked) > 0:
            print("mnemonic" + str((checked)))
            results.append(str.upper(checked[0]))
        elif len(checked2) > 0:
            print("section" + str((checked2)))
        else:
            print("other[" + s_line)

# 保存する.
with open("memo\\results.json", "w") as f:
    json.dump(results, f, indent=4)
