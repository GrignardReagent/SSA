nohup python IY014_simulation.py > IY014_simulation.out
sleep 60
nohup python IY014_simulation_cv.py > IY014_simulation_cv.out
sleep 60
nohup python IY014_simulation_t_ac.py > IY014_simulation_t_ac.out
sleep 60
echo "All simulations completed."