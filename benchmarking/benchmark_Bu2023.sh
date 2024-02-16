export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
svdmodel-benchmark \
        --model Bu2023Ye \
        --ncpus 1 \
        --outdir outdir_Bu2023 \
        --data-path /home/urash/twouters/KN_Lightcurves/lightcurves/bulla_2023 \
        --interpolation-type tensorflow \
        --svd-path /home/urash/twouters/new_nmma_models/ \
        --local-only