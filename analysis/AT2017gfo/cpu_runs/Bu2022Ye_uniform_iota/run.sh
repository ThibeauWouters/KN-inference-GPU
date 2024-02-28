export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

model='Bu2022Ye'
iotatype='uniform'

lightcurve-analysis \
    --model ${model} \
    --svd-path /home/urash/twouters/nmma_models \
    --outdir outdir \
    --label AT2017gfo_${model}_${iotatype}_iota \
    --trigger-time 57982.5285236896 \
    --data ../AT2017gfo.dat \
    --prior ../priors/AT2017gfo_${model}_${iotatype}_iota.prior \
    --tmin 0.05 \
    --tmax 14 \
    --dt 0.1 \
    --error-budget 1 \
    --nlive 2048 \
    --Ebv-max 0 \
    --interpolation-type tensorflow \
    --plot
