from lib2to3.pytree import convert
import otf2
import re

import copy

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

value_dic = {"_": [1]}
value_dic.clear()
diff_dic = {"_": [1]}
diff_dic.clear()
print(type(otf2))
with otf2.reader.open('main.otf2.otf2') as trace:
    # trace.definitions._set_clock_properties()
    print("Read {} string definitions".format(len(trace.definitions.strings)))
    print((trace.definitions.strings[1]))
    # for name in trace.definitions.strings:
    #     print("name:{}".format(name))

    for location, event in trace.events:
        print(event.time, type(event).__name__, (event.metric.members[0].name if (type(event).__name__== "Metric") else "none"), sep=",")
        if type(event).__name__ == "Metric":
            event_name = event.metric.members[0].name
            time = event.time
            value = event.values[0]
            # print(event_name,",")
            if (re.match("PAPI_.+", event_name)):
                # エラー回避用
                if not (event_name in diff_dic.keys()):
                    # init list
                    value_dic[event_name] = [value]
                    diff_dic[event_name] = [value]
                # 差分計算
                diff_value = value - diff_dic[event_name][-1]
                

                # print("[E={:^15}] time={:0=10}, value={:06}, delta={:0=+7}".format(
                #     event_name,
                #     time, value, diff_value))
                # 前回を記録
                if value != 0 and value!=1: # 最後にある謎の値を削除
                    value_dic[event_name].append(value)
                    diff_dic[event_name].append(diff_value)
print("")

# convert list to Series
###
pd_list = {"_": pd.Series([1])}
pd_list.clear()
for key,value in value_dic.items():
    print(key)
    pd_list[key] = pd.Series(value)
    pd_list[key].plot(label=key)
    plt.legend()
    plt.savefig('dst/img/plot_'+key+'.png')
    plt.close('all')
# plt.legend()
plt.savefig('dst/img/plot_ALL.png')
plt.close('all')

plt.figure()
pd_list.clear()
for key,value in diff_dic.items():
    pd_list[key] = pd.Series(value)
    pd_list[key].plot(label=key+"_diff")
    plt.legend()
    plt.savefig('dst/img/plot_'+key+'_diff.png')
    plt.close('all')
# plt.legend()
plt.savefig('dst/img/plot_ALL_diff.png')
plt.close('all')
