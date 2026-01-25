nohup python IY015_baseline_model_training_t_ac.py > IY015_baseline_model_training_t_ac.out 
sleep 60
nohup python IY015_baseline_model_training_mu.py > IY015_baseline_model_training_mu.out
sleep 60
nohup python IY015_baseline_model_training_cv.py > IY015_baseline_model_training_cv.out
sleep 60
nohup python IY015_baseline_model_training.py > IY015_baseline_model_training_1.out 
sleep 60
echo "All training scripts launched."