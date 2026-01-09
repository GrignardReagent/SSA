nohup python IY011_simulation_t_ac.py > IY011_simulation_t_ac_1.out
sleep 60
nohup python IY011_simulation_cv.py > IY011_simulation_cv_1.out
sleep 60    
nohup python IY011_simulation_mu.py > IY011_simulation_mu_1.out
echo "All simulation jobs completed."