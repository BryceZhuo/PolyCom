export DARA_PATH=${Your data path}
export OUTPUT_PATH=${Your output path}


cd examples/llama2
sh run_pretrain_megatron_llama.sh \
dsw  \
../../ \
1.3B   \
2    \
64 \
3e-4   \
3e-5   \
4096 \
4096  \
0   \
bf16  \
1   \
1  \
sel  \
true   \
true  \
true \
false   \
1000  \
${DARA_PATH}  \
none   \
100000000000  \
620000000   \
${OUTPUT_PATH} \
polyrelu