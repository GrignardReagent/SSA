# run all simulations then email myself when done

nohup python IY019_simulation.py > IY019_simulation.out
echo "Simulations complete. Check IY019_simulation.out for details." | mail -s "IY019 Simulations Finished on $(hostname)" ian.yang@ed.ac.uk

nohup python IY019_simulation_mu.py > IY019_simulation_mu.out
echo "Simulations with mu variation complete. Check IY019_simulation_mu.out for details." | mail -s "IY019 Mu Variation Simulations Finished on $(hostname)" ian.yang@ed.ac.uk

nohup python IY019_simulation_cv.py > IY019_simulation_cv.out 
echo "IY019 CV variation simulations complete. Check IY019_simulation_cv.out for details." | mail -s "IY019 CV Variation Simulations Finished on $(hostname)" ian.yang@ed.ac.uk

nohup python IY019_simulation_t_ac.py > IY019_simulation_t_ac.out 
echo "IY019 t_ac variation simulations complete. Check IY019_simulation_t_ac.out for details." | mail -s "IY019 t_ac Variation Simulations Finished on $(hostname)" ian.yang@ed.ac.uk