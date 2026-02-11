#python IY020_simulation_mu.py > IY020_simulation_mu.out
nohup python IY020_simulation.py > IY020_simulation.out
sleep 60
nohup python IY020_simulation_cv.py > IY020_simulation_cv.out
sleep 60
nohup python IY020_simulation_t_ac.py > IY020_simulation_t_ac.out
sleep 60
echo "All simulations completed."