#
rm main.dump tautrace.0.0.0.trc events.0.edf a.out
rm -r MULTI__*/

# read -p "run? (y/N): " yn
# case "$yn" in [yY]*) ;; *) echo "abort." ; exit ;; esac

export TAU_TRACE=1
# export TAU_PROFILE=1
export COUNTER1=P_WALL_CLOCK_TIME

# before exec python
mkdir dst
mkdir dst/img

build()
{
    gcc main.c -S
    taucc main.c -tau:pdtinst -tau:serial,papi,pdt
    echo "10" | ./a.out

    # read -p "plz Enter: "

    tau_convert -dump ./tautrace.0.0.0.trc ./events.0.edf > main.dump
    tau_trace2json ./tautrace.0.0.0.trc ./events.0.edf -chrome -ignoreatomic -o trace.json
    tau2otf2 ./tautrace.0.0.0.trc ./events.0.edf main.otf2
    # tau2profile ./tautrace.0.0.0.trc ./events.0.edf -d ./

    python open_otf2.py
    #-ignoreatomic
    # tail -n 10 main.dump
}


build
#GET_TIME_OF_DAY
# export COUNTER2=LINUX_TIMERS
# export COUNTER2=PAPI_L1_DCM
# export COUNTER3=PAPI_L1_ICM
# export COUNTER4=PAPI_L2_DCM
# export COUNTER5=PAPI_L2_ICM
# build
# export COUNTER2=PAPI_L1_TCM
# export COUNTER3=PAPI_L2_TCM
# export COUNTER4=PAPI_L3_TCM
# export COUNTER5=PAPI_CA_SNP
# build
# export COUNTER2=PAPI_CA_SHR
# export COUNTER3=PAPI_CA_CLN
# export COUNTER4=PAPI_CA_INV
# export COUNTER5=PAPI_CA_ITV
# build
# export COUNTER2=PAPI_L3_LDM
# export COUNTER3=PAPI_TLB_DM
# export COUNTER4=PAPI_TLB_IM
# export COUNTER5=PAPI_L1_LDM
# build
# export COUNTER2=PAPI_L1_STM
# export COUNTER3=PAPI_L2_LDM
# export COUNTER4=PAPI_L2_STM
# export COUNTER5=PAPI_PRF_DM
# build
# export COUNTER2=PAPI_MEM_WCY
# export COUNTER3=PAPI_STL_ICY
# export COUNTER4=PAPI_FUL_ICY
# export COUNTER5=PAPI_STL_CCY
# build
# export COUNTER2=PAPI_FUL_CCY
# export COUNTER3=PAPI_BR_UCN
# export COUNTER4=PAPI_BR_CN
# export COUNTER5=PAPI_BR_TKN
# build
# export COUNTER2=PAPI_BR_NTK
# export COUNTER3=PAPI_BR_MSP
# export COUNTER4=PAPI_BR_PRC
# export COUNTER5=PAPI_TOT_INS
# build
# export COUNTER2=PAPI_LD_INS
# export COUNTER3=PAPI_SR_INS
# export COUNTER4=PAPI_BR_INS
# export COUNTER5=PAPI_RES_STL
# build
# export COUNTER2=PAPI_TOT_CYC
# export COUNTER3=PAPI_LST_INS
# export COUNTER4=PAPI_L2_DCA
# export COUNTER5=PAPI_L3_DCA
# build
# export COUNTER2=PAPI_L2_DCR
# export COUNTER3=PAPI_L3_DCR
# export COUNTER4=PAPI_L2_DCW
# export COUNTER5=PAPI_L3_DCW
# build
# export COUNTER2=PAPI_L2_ICH
# export COUNTER3=PAPI_L2_ICA
# export COUNTER4=PAPI_L3_ICA
# export COUNTER5=PAPI_L2_ICR
# build
# export COUNTER2=PAPI_L3_ICR
# export COUNTER3=PAPI_L2_TCA
# export COUNTER4=PAPI_L3_TCA
# export COUNTER5=PAPI_L2_TCR
# build
# export COUNTER2=PAPI_L3_TCR
# export COUNTER3=PAPI_L2_TCW
# export COUNTER4=PAPI_L3_TCW
# export COUNTER5=PAPI_REF_CYC
# build
