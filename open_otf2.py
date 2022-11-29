from lib2to3.pytree import convert
import otf2
import re

import copy

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import json

value_dic = {"_": [1]}
value_dic.clear()
diff_dic = {"_": [1]}
diff_dic.clear()
stack = {"_main": [
    {"name": "main", "callStack": 0, "events": [
        {"name": "PAPI_", "start": 100, "end": 105, "diff": 5},
    ]}
]}
stack.clear()  # 初期化
stack_done = []

ptr_target = {"name": "name", "callStack": 0, "events": []}
""" 現在のターゲット関数へのポインター代わり """
ptr_target = None  # 初期化


def countFunc(arr: tuple, funcName: str) -> None:
    """
    Args:
        arr (tuple): a
        funcName (str): b
    """
    filter(lambda v: v["name"] == funcName, arr)


print(type(otf2))
with otf2.reader.open('main.otf2.otf2') as trace:
    # trace.definitions._set_clock_properties()
    print("Read {} string definitions".format(len(trace.definitions.strings)))
    print((trace.definitions.strings[1]))
    # for name in trace.definitions.strings:
    #     print("name:{}".format(name))

    for location, event in trace.events:
        print(event.time, type(event).__name__, ((event.metric.members[0].name+":"+str(event.values[0])+":"+str(
            len(event.values))) if (type(event).__name__ == "Metric") else "none"), sep=",")
        if type(event).__name__ == "Metric":
            if ptr_target is None:
                break
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

                # switch
                if(ptr_target["state"] == "Enter"):
                    _obj = {"name": event_name, "start": value, "end": -1}
                    ptr_target["events"].append(_obj)
                    print()
                elif(ptr_target["state"] == "Leave"):
                    _targetEvent = list(filter(
                        lambda v: v["name"] == event_name, ptr_target["events"]))[0]
                    _targetEvent["end"] = value
                    print()

                # print("[E={:^15}] time={:0=10}, value={:06}, delta={:0=+7}".format(
                #     event_name,
                #     time, value, diff_value))
                # 前回を記録
                if value != 0 and value != 1:  # 最後にある謎の値を削除
                    value_dic[event_name].append(value)
                    diff_dic[event_name].append(diff_value)
        elif type(event).__name__ == "Enter":
            _func_name = event.region.canonical_name
            if not (_func_name in stack.keys()):
                stack[_func_name] = []
            _nest = len(stack[_func_name])
            ptr_target = {"name": _func_name, "callStack": _nest,
                          "events": [], "state": "Enter"}
            stack[_func_name].append(ptr_target)
            print("Enter {},{}".format(event.region, event.attributes))
        elif type(event).__name__ == "Leave":
            _func_name = event.region.canonical_name
            #  stack[_func_name][-1]
            # filter(lambda v:v["s"], stack[_func_name])
            ptr_target = stack[_func_name].pop()
            ptr_target["state"] = "Leave"
            stack_done.append(ptr_target)  # とりあえず退避する. 可能ならもっとましな所で実行したいが...
            print("Leave {},{}".format(event.region, event.attributes))
print(json.dumps(stack_done))
print("")

# convert list to Series
###
pd_list = {"_": pd.Series([1])}
pd_list.clear()
for key, value in value_dic.items():
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
for key, value in diff_dic.items():
    pd_list[key] = pd.Series(value)
    pd_list[key].plot(label=key+"_diff")
    plt.legend()
    plt.savefig('dst/img/plot_'+key+'_diff.png')
    plt.close('all')
# plt.legend()
plt.savefig('dst/img/plot_ALL_diff.png')
plt.close('all')

"""
.TAU A 100
main A 95
calc A 80
"""
