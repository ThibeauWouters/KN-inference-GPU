export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

mpiexec -np 16 lightcurve-analysis \
    --model Bu2022Ye \
    --svd-path /home/urash/twouters/flax_models \
    --outdir outdir \
    --label AT2017gfo_corrected \
    --trigger-time 57982.5285236896 \
    --data ../data/AT2017gfo_corrected.dat \
    --prior ../priors/AT2017gfo.prior \
    --tmin 0.05 \
    --tmax 14 \
    --dt 0.1 \
    --error-budget 1 \
    --nlive 2048 \
    --Ebv-max 0 \
    --local-only \
    --interpolation-type flax \
    --plot