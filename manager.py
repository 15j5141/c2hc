import subprocess, os


class Manager:
    def buildC(self, *args, **kwargs):
        HCList = [
            "PAPI_L1_DCM",
            "PAPI_L1_ICM",
            "PAPI_L2_DCM",
            "PAPI_L2_ICM",
            "PAPI_L1_TCM",
            "PAPI_L2_TCM",
            "PAPI_L3_TCM",
            "PAPI_CA_SNP",
            "PAPI_CA_SHR",
            "PAPI_CA_CLN",
            "PAPI_CA_INV",
            "PAPI_CA_ITV",
            "PAPI_L3_LDM",
            "PAPI_TLB_DM",
            "PAPI_TLB_IM",
            "PAPI_L1_LDM",
            "PAPI_L1_STM",
            "PAPI_L2_LDM",
            "PAPI_L2_STM",
            "PAPI_PRF_DM",
            "PAPI_MEM_WCY",
            "PAPI_STL_ICY",
            "PAPI_FUL_ICY",
            "PAPI_STL_CCY",
            "PAPI_FUL_CCY",
            "PAPI_BR_UCN",
            "PAPI_BR_CN",
            "PAPI_BR_TKN",
            "PAPI_BR_NTK",
            "PAPI_BR_MSP",
            "PAPI_BR_PRC",
            "PAPI_TOT_INS",
            "PAPI_LD_INS",
            "PAPI_SR_INS",
            "PAPI_BR_INS",
            "PAPI_RES_STL",
            "PAPI_TOT_CYC",
            "PAPI_LST_INS",
            "PAPI_L2_DCA",
            "PAPI_L3_DCA",
            "PAPI_L2_DCR",
            "PAPI_L3_DCR",
            "PAPI_L2_DCW",
            "PAPI_L3_DCW",
            "PAPI_L2_ICH",
            "PAPI_L2_ICA",
            "PAPI_L3_ICA",
            "PAPI_L2_ICR",
            "PAPI_L3_ICR",
            "PAPI_L2_TCA",
            "PAPI_L3_TCA",
            "PAPI_L2_TCR",
            "PAPI_L3_TCR",
            "PAPI_L2_TCW",
            "PAPI_L3_TCW",
            "PAPI_REF_CYC",
        ]
        i = 0
        my_env = os.environ.copy()
        for i in range(len(HCList / 4)):

            my_env["COUNTER2"] = HCList[4 * i + 0]
            my_env["COUNTER3"] = HCList[4 * i + 1]
            my_env["COUNTER4"] = HCList[4 * i + 2]
            my_env["COUNTER5"] = HCList[4 * i + 3]
            self.called_build()

    def called_build(self, *args, **kwargs):
        process = subprocess.Popen(
            ["sh", "./called_build.sh"],
            stdout=subprocess.PIPE,
            env=my_env,
            encoding="utf-8",
        )
        # process = subprocess.Popen(["cmd", "/c", "echo", "%MSG%"], stdout=subprocess.PIPE, env=my_env, encoding="sjis")
        stdoutdata, _ = process.communicate()

        print(stdoutdata)
        print(process.returncode)
