from lib2to3.pytree import convert
import otf2
import re

import copy

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import json
import csv
import os

class OpenOTF2:
    def __init__(self):
        self.stack = {
            "_main": [
                {
                    "name": "main",
                    "callStack": 0,
                    "events": [
                        {"name": "PAPI_", "start": 100, "end": 105, "diff": 5},
                    ],
                }
            ]
        }
        self.stack.clear()  # 初期化
        self.stack_done = []

        self.ptr_target = {"name": "name", "callStack": 0, "events": []}
        """ 現在のターゲット関数へのポインター代わり """
        self.ptr_target = None  # 初期化

    def countFunc(arr: tuple, funcName: str) -> None:
        """
        Args:
            arr (tuple): a
            funcName (str): b
        """
        filter(lambda v: v["name"] == funcName, arr)

    def load(self, filename="main.otf2.otf2"):
        with otf2.reader.open(filename) as trace:
            # trace.definitions._set_clock_properties()
            # print("Read {} string definitions".format(len(trace.definitions.strings)))
            # print((trace.definitions.strings[1]))

            for location, event in trace.events:
                if type(event).__name__ == "Metric":
                    if self.ptr_target is None:
                        break
                    event_name = event.metric.members[0].name
                    time = event.time
                    value = event.values[0]
                    # print(event_name,",")
                    if re.match("PAPI_.+", event_name):

                        # switch
                        if self.ptr_target["state"] == "Enter":
                            _obj = {"name": event_name, "start": value, "end": -1, "diff": -1}
                            self.ptr_target["events"].append(_obj)
                        elif self.ptr_target["state"] == "Leave":
                            _targetEvent = list(
                                filter(lambda v: v["name"] == event_name, self.ptr_target["events"])
                            )[0]
                            _targetEvent["end"] = value
                            _targetEvent["diff"] = value - _targetEvent["start"]
                            # diff

                        # print("[E={:^15}] time={:0=10}, value={:06}, delta={:0=+7}".format(
                        #     event_name,
                        #     time, value, diff_value))
                elif type(event).__name__ == "Enter":
                    _func_name = event.region.canonical_name
                    if not (_func_name in self.stack.keys()):
                        self.stack[_func_name] = []
                    _nest = len(self.stack[_func_name])
                    self.ptr_target = {
                        "name": _func_name,
                        "callStack": _nest,
                        "events": [],
                        "state": "Enter",
                    }
                    self.stack[_func_name].append(self.ptr_target)
                    # print("Enter {},{}".format(event.region, event.attributes))
                elif type(event).__name__ == "Leave":
                    _func_name = event.region.canonical_name
                    #  self.stack[_func_name][-1]
                    # filter(lambda v:v["s"], self.stack[_func_name])
                    self.ptr_target = self.stack[_func_name].pop()
                    self.ptr_target["state"] = "Leave"
                    self.stack_done.append(self.ptr_target)  # とりあえず退避する. 可能ならもっとましな所で実行したいが...
                    # print("Leave {},{}".format(event.region, event.attributes))
    def save(self):
        # print(json.dumps(stack_done))
        with open("out/hardwareCounter.json", "w") as f:
            json.dump(self.stack_done, f, indent=2)

        # row = [x["diff"] for x in self.stack_done[0]["events"] ]
        row = [x["diff"] for x in self.stack_done[len(self.stack_done)-2]["events"] ]
        print("done: open_otf2")

        with open('out/hc_append.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(row)

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__)) + "/"
    reader = OpenOTF2()
    reader.load()
    reader.save()

if __name__ == "__main__":
    main()