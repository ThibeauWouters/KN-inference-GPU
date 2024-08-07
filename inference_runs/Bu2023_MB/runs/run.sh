export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

### User settings here:
export prior_filename=AT2017gfo_dL44 # choose prior, also used as label
no_inf_data=false # whether or not to include the non-detections

# Check which dataset to use, which is also used as the label for the run
if [ "$no_inf_data" = true ]; then
    data_file="../data/AT2017gfo_corrected_no_inf.dat"
    label="${prior_filename}_no_inf"
else
    data_file="../data/AT2017gfo_corrected.dat"
    label="${prior_filename}"
fi

mpiexec -np 16 lightcurve-analysis \
    --model Bu2023Ye \
    --svd-path /home/urash/twouters/new_nmma_models/Bu2023Ye_TW/ \
    --outdir "./outdir/" \
    --label $label \
    --trigger-time 57982.5285236896 \
    --data $data_file \
    --prior ../priors/$prior_filename.prior \
    --tmin 0.05 \
    --tmax 14 \
    --dt 0.1 \
    --error-budget 1 \
    --nlive 2048 \
    --Ebv-max 0 \
    --interpolation-type tensorflow \
    --local-only \
    --plot
