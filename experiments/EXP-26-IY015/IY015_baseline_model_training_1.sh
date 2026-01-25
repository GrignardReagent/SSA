nohup python IY015_baseline_model_training_t_ac.py > IY015_baseline_model_training_t_ac_1.out 
sleep 60
nohup python IY015_baseline_model_training_mu.py > IY015_baseline_model_training_mu_1.out
sleep 60
nohup python IY015_baseline_model_training_cv.py > IY015_baseline_model_training_cv_1.out
sleep 60
echo "All training scripts launched."