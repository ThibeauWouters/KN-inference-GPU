export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
svdmodel-benchmark \
        --model Bu2022Ye \
        --ncpus 1 \
        --outdir outdir_Bu2022 \
        --data-path /home/urash/twouters/KN_Lightcurves/lightcurves/bulla_2022 \
        --interpolation-type tensorflow \
        --svd-path /home/urash/twouters/nmma_models/Bu2022Ye_tf \