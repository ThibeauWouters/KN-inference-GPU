export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

### User settings here:
export prior_filename=AT2017gfo_dL40 # choose prior, also used as label
no_inf_data=false
use_old_prior=false

# Check which dataset
if [ "$no_inf_data" = true ]; then
    label="${prior_filename}_no_inf"
    data_file="../data/AT2017gfo_corrected_no_inf.dat"
else
    data_file="../data/AT2017gfo_corrected.dat"
    label="${prior_filename}"
fi

# Check which prior
if [ "$use_old_prior" = true ]; then
    prior_dir="old_priors"
    # outdir="outdir_old_prior"
else
    prior_dir="new_priors"
    # outdir="outdir"
fi

mpiexec -np 16 lightcurve-analysis \
    --model Bu2023Ye \
    --svd-path /home/urash/twouters/new_nmma_models/old_Bu2023Ye_tf \
    --outdir "./outdir_old_NMMA/" \
    --label $label \
    --trigger-time 57982.5285236896 \
    --data $data_file \
    --prior ../$prior_dir/$prior_filename.prior \
    --tmin 0.05 \
    --tmax 14 \
    --dt 0.1 \
    --error-budget 1 \
    --nlive 2048 \
    --Ebv-max 0 \
    --interpolation-type tensorflow \
    --local-only \
    --plot
