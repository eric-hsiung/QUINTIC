#!/bin/bash

## To disable counterexample-guided expansion, remove the --enable-cex-expansion option.

## Classification CC-VE
for TEST in test_zero_A test_zero_B test_one_A test_one_A_extra test_one_B test_two_C testB testB2 testB3 testB4 testB5 testB6 testC_linear simple_rm_A simple_rm_B1 simple_rm_B2 simple_rm_B3 simple_rm_B4 simple_rm_B5 simple_rm_B6 simple_rm_B7 simple_office_t4_A simple_office_t4_B simple_office_t4_C simple_craft_t105_A;
do
for j in `seq 0 99`;
do
    sem --id quintic --bg --jobs 20 "nohup python experiment_sum_SMT.py --experiment-type moore --test-name $TEST --num-trials 1 --enable-cex-expansion --feedback-strength strong --trial $j --valuation-model classification 1> /dev/null 2> logs/$TEST.err.$j "
done
done

for TEST in test_zero_A test_zero_B test_one_A test_one_A_extra test_one_B test_two_C testB testB2 testB3 testB4 testB5 testB6 testC_linear simple_rm_A simple_rm_B1 simple_rm_B2 simple_rm_B3 simple_rm_B4 simple_rm_B5 simple_rm_B6 simple_rm_B7 simple_office_t4_A simple_office_t4_B simple_office_t4_C simple_craft_t105_A;
do
for j in `seq 0 99`;
do
    sem --id quintic --bg --jobs 20 "nohup python experiment_sum_SMT.py --experiment-type moore --test-name $TEST --num-trials 1 --enable-cex-expansion --feedback-strength weak --trial $j --valuation-model classification 1> /dev/null 2> logs/$TEST.err.$j "
done
done

## Summation CC-VE
for TEST in test_zero_A test_zero_B test_one_A test_one_A_extra test_one_B test_two_C testB testB2 testB3 testB4 testB5 testB6 testC_linear simple_rm_A simple_rm_B1 simple_rm_B2 simple_rm_B3 simple_rm_B4 simple_rm_B5 simple_rm_B6 simple_rm_B7 simple_office_t4_A simple_office_t4_B simple_office_t4_C simple_craft_t105_A;
do
for j in `seq 0 99`;
do
    sem --id quintic --bg --jobs 20 "nohup python experiment_sum_SMT.py --experiment-type moore --test-name $TEST --num-trials 1 --enable-cex-expansion --feedback-strength strong --trial $j --valuation-model sum 1> /dev/null 2> logs/$TEST.err.$j "
done
done

for TEST in test_zero_A test_zero_B test_one_A test_one_A_extra test_one_B test_two_C testB testB2 testB3 testB4 testB5 testB6 testC_linear simple_rm_A simple_rm_B1 simple_rm_B2 simple_rm_B3 simple_rm_B4 simple_rm_B5 simple_rm_B6 simple_rm_B7 simple_office_t4_A simple_office_t4_B simple_office_t4_C simple_craft_t105_A;
do
for j in `seq 0 99`;
do
    sem --id quintic --bg --jobs 20 "nohup python experiment_sum_SMT.py --experiment-type moore --test-name $TEST --num-trials 1 --enable-cex-expansion --feedback-strength weak --trial $j --valuation-model sum 1> /dev/null 2> logs/$TEST.err.$j "
done
done

## Discounted Summation CC-VE
for TEST in test_zero_A test_zero_B test_one_A test_one_A_extra test_one_B test_two_C testB testB2 testB3 testB4 testB5 testB6 testC_linear simple_rm_A simple_rm_B1 simple_rm_B2 simple_rm_B3 simple_rm_B4 simple_rm_B5 simple_rm_B6 simple_rm_B7 simple_office_t4_A simple_office_t4_B simple_office_t4_C simple_craft_t105_A;
do
for j in `seq 0 99`;
do
    sem --id quintic --bg --jobs 20 "nohup python experiment_sum_SMT.py --experiment-type moore --test-name $TEST --num-trials 1 --enable-cex-expansion --feedback-strength strong --trial $j --valuation-model discountsum --gamma 0.99 1> /dev/null 2> logs/$TEST.err.$j "
done
done

for TEST in test_zero_A test_zero_B test_one_A test_one_A_extra test_one_B test_two_C testB testB2 testB3 testB4 testB5 testB6 testC_linear simple_rm_A simple_rm_B1 simple_rm_B2 simple_rm_B3 simple_rm_B4 simple_rm_B5 simple_rm_B6 simple_rm_B7 simple_office_t4_A simple_office_t4_B simple_office_t4_C simple_craft_t105_A;
do
for j in `seq 0 99`;
do
    sem --id quintic --bg --jobs 20 "nohup python experiment_sum_SMT.py --experiment-type moore --test-name $TEST --num-trials 1 --enable-cex-expansion --feedback-strength weak --trial $j --valuation-model discountsum --gamma 0.99 1> /dev/null 2> logs/$TEST.err.$j "
done
done


## Product CC-VE
for TEST in test_zero_A test_zero_B test_one_A test_one_A_extra test_one_B test_two_C testB testB2 testB3 testB4 testB5 testB6 testC_linear simple_rm_A simple_rm_B1 simple_rm_B2 simple_rm_B3 simple_rm_B4 simple_rm_B5 simple_rm_B6 simple_rm_B7 simple_office_t4_A simple_office_t4_B simple_office_t4_C simple_craft_t105_A;
do
for j in `seq 0 99`;
do
    sem --id quintic --bg --jobs 20 "nohup python experiment_sum_SMT.py --experiment-type moore --test-name $TEST --num-trials 1 --enable-cex-expansion --feedback-strength strong --trial $j --valuation-model prod 1> /dev/null 2> logs/$TEST.err.$j "
done
done

for TEST in test_zero_A test_zero_B test_one_A test_one_A_extra test_one_B test_two_C testB testB2 testB3 testB4 testB5 testB6 testC_linear simple_rm_A simple_rm_B1 simple_rm_B2 simple_rm_B3 simple_rm_B4 simple_rm_B5 simple_rm_B6 simple_rm_B7 simple_office_t4_A simple_office_t4_B simple_office_t4_C simple_craft_t105_A;
do
for j in `seq 0 99`;
do
    sem --id quintic --bg --jobs 20 "nohup python experiment_sum_SMT.py --experiment-type moore --test-name $TEST --num-trials 1 --enable-cex-expansion --feedback-strength weak --trial $j --valuation-model prod 1> /dev/null 2> logs/$TEST.err.$j "
done
done


## Classification VE
for TEST in test_zero_A test_zero_B test_one_A test_one_A_extra test_one_B test_two_C testB testB2 testB3 testB4 testB5 testB6 testC_linear simple_rm_A simple_rm_B1 simple_rm_B2 simple_rm_B3 simple_rm_B4 simple_rm_B5 simple_rm_B6 simple_rm_B7 simple_office_t4_A simple_office_t4_B simple_office_t4_C simple_craft_t105_A;
do
for j in `seq 0 99`;
do
    sem --id quintic --bg --jobs 20 "nohup python experiment_sum_SMT.py --experiment-type moore --test-name $TEST --num-trials 1 --enable-cex-expansion --feedback-strength strong --disable-cc-obj --trial $j --valuation-model classification 1> /dev/null 2> logs/$TEST.err.$j "
done
done

for TEST in test_zero_A test_zero_B test_one_A test_one_A_extra test_one_B test_two_C testB testB2 testB3 testB4 testB5 testB6 testC_linear simple_rm_A simple_rm_B1 simple_rm_B2 simple_rm_B3 simple_rm_B4 simple_rm_B5 simple_rm_B6 simple_rm_B7 simple_office_t4_A simple_office_t4_B simple_office_t4_C simple_craft_t105_A;
do
for j in `seq 0 99`;
do
    sem --id quintic --bg --jobs 20 "nohup python experiment_sum_SMT.py --experiment-type moore --test-name $TEST --num-trials 1 --enable-cex-expansion --feedback-strength weak --disable-cc-obj --trial $j --valuation-model classification 1> /dev/null 2> logs/$TEST.err.$j "
done
done

## Summation VE
for TEST in test_zero_A test_zero_B test_one_A test_one_A_extra test_one_B test_two_C testB testB2 testB3 testB4 testB5 testB6 testC_linear simple_rm_A simple_rm_B1 simple_rm_B2 simple_rm_B3 simple_rm_B4 simple_rm_B5 simple_rm_B6 simple_rm_B7 simple_office_t4_A simple_office_t4_B simple_office_t4_C simple_craft_t105_A;
do
for j in `seq 0 99`;
do
    sem --id quintic --bg --jobs 20 "nohup python experiment_sum_SMT.py --experiment-type moore --test-name $TEST --num-trials 1 --enable-cex-expansion --feedback-strength strong --disable-cc-obj --trial $j --valuation-model sum 1> /dev/null 2> logs/$TEST.err.$j "
done
done

for TEST in test_zero_A test_zero_B test_one_A test_one_A_extra test_one_B test_two_C testB testB2 testB3 testB4 testB5 testB6 testC_linear simple_rm_A simple_rm_B1 simple_rm_B2 simple_rm_B3 simple_rm_B4 simple_rm_B5 simple_rm_B6 simple_rm_B7 simple_office_t4_A simple_office_t4_B simple_office_t4_C simple_craft_t105_A;
do
for j in `seq 0 99`;
do
    sem --id quintic --bg --jobs 20 "nohup python experiment_sum_SMT.py --experiment-type moore --test-name $TEST --num-trials 1 --enable-cex-expansion --feedback-strength weak --disable-cc-obj --trial $j --valuation-model sum 1> /dev/null 2> logs/$TEST.err.$j "
done
done

## Discounted Summation VE
for TEST in test_zero_A test_zero_B test_one_A test_one_A_extra test_one_B test_two_C testB testB2 testB3 testB4 testB5 testB6 testC_linear simple_rm_A simple_rm_B1 simple_rm_B2 simple_rm_B3 simple_rm_B4 simple_rm_B5 simple_rm_B6 simple_rm_B7 simple_office_t4_A simple_office_t4_B simple_office_t4_C simple_craft_t105_A;
do
for j in `seq 0 99`;
do
    sem --id quintic --bg --jobs 20 "nohup python experiment_sum_SMT.py --experiment-type moore --test-name $TEST --num-trials 1 --enable-cex-expansion --feedback-strength strong --disable-cc-obj --trial $j --valuation-model discountsum --gamma 0.99 1> /dev/null 2> logs/$TEST.err.$j "
done
done

for TEST in test_zero_A test_zero_B test_one_A test_one_A_extra test_one_B test_two_C testB testB2 testB3 testB4 testB5 testB6 testC_linear simple_rm_A simple_rm_B1 simple_rm_B2 simple_rm_B3 simple_rm_B4 simple_rm_B5 simple_rm_B6 simple_rm_B7 simple_office_t4_A simple_office_t4_B simple_office_t4_C simple_craft_t105_A;
do
for j in `seq 0 99`;
do
    sem --id quintic --bg --jobs 20 "nohup python experiment_sum_SMT.py --experiment-type moore --test-name $TEST --num-trials 1 --enable-cex-expansion --feedback-strength weak --disable-cc-obj --trial $j --valuation-model discountsum --gamma 0.99 1> /dev/null 2> logs/$TEST.err.$j "
done
done


## Product VE
for TEST in test_zero_A test_zero_B test_one_A test_one_A_extra test_one_B test_two_C testB testB2 testB3 testB4 testB5 testB6 testC_linear simple_rm_A simple_rm_B1 simple_rm_B2 simple_rm_B3 simple_rm_B4 simple_rm_B5 simple_rm_B6 simple_rm_B7 simple_office_t4_A simple_office_t4_B simple_office_t4_C simple_craft_t105_A;
do
for j in `seq 0 99`;
do
    sem --id quintic --bg --jobs 20 "nohup python experiment_sum_SMT.py --experiment-type moore --test-name $TEST --num-trials 1 --enable-cex-expansion --feedback-strength strong --disable-cc-obj --trial $j --valuation-model prod 1> /dev/null 2> logs/$TEST.err.$j "
done
done

for TEST in test_zero_A test_zero_B test_one_A test_one_A_extra test_one_B test_two_C testB testB2 testB3 testB4 testB5 testB6 testC_linear simple_rm_A simple_rm_B1 simple_rm_B2 simple_rm_B3 simple_rm_B4 simple_rm_B5 simple_rm_B6 simple_rm_B7 simple_office_t4_A simple_office_t4_B simple_office_t4_C simple_craft_t105_A;
do
for j in `seq 0 99`;
do
    sem --id quintic --bg --jobs 20 "nohup python experiment_sum_SMT.py --experiment-type moore --test-name $TEST --num-trials 1 --enable-cex-expansion --feedback-strength weak --disable-cc-obj --trial $j --valuation-model prod 1> /dev/null 2> logs/$TEST.err.$j "
done
done


