nohup python IY014_baseline_model_training_cv.py > IY014_baseline_model_training_cv.out
sleep 60
nohup python IY014_baseline_model_training_mu.py > IY014_baseline_model_training_mu.out
sleep 60
nohup python IY014_baseline_model_training_t_ac.py > IY014_baseline_model_training_t_ac.out
echo "CV simulations and model trainings for EXP-26-IY014 have been completed."