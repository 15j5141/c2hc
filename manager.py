import subprocess, os
my_env = os.environ.copy()
my_env["COUNTER2"] = "PAPI_L1_DCM"
my_env["COUNTER3"] = "PAPI_L1_ICM"
my_env["COUNTER4"] = "PAPI_L2_DCM"
my_env["COUNTER5"] = "PAPI_L2_ICM"

process = subprocess.Popen(["sh", "./called_build.sh"], stdout=subprocess.PIPE, env=my_env, encoding="utf-8")
# process = subprocess.Popen(["cmd", "/c", "echo", "%MSG%"], stdout=subprocess.PIPE, env=my_env, encoding="sjis")
stdoutdata, _ = process.communicate()

print(stdoutdata)
print(process.returncode)
