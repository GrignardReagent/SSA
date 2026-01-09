echo "Starting variation training jobs..."
nohup python IY011_baseline_model_training.py > IY011_baseline_model_training_7.out
sleep 60
nohup python IY011_baseline_model_training_t_ac.py > IY011_baseline_model_training_7_t_ac.out
sleep 60
nohup python IY011_baseline_model_training_cv.py > IY011_baseline_model_training_7_cv.out
sleep 60
nohup python IY011_baseline_model_training_mu.py > IY011_baseline_model_training_7_mu.out
echo "All training jobs completed."