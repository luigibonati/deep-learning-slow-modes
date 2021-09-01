# GROMACS SET-UP HELPER
set -e
# adapted from Sandro Bottaro

#############################################
# Step 01 - PDB2GMX_MPI = choose CHARMM27 and TIP3P 
#echo -e "8\n1" |
gmx_mpi pdb2gmx -f frame.pdb -ignh -o conf_01.gro -p topol_01.top -i posre_01.itp

# Step 02 - Prepare for vacuum minimization
gmx_mpi grompp -f mini.mdp -c conf_01.gro -p topol_01.top -po mdp_02.mdp -o tpr_mini.tpr

# Step 03 - Run minimization in vacuum
gmx_mpi mdrun -s tpr_mini.tpr -deffnm mini

#############################################
# Step 04 - Define simulation box
#gmx_mpi editconf -bt dodecahedron -d 1.15 -f mini.gro -o conf_04.gro

# Step 05 - Add Water molecules 
gmx_mpi solvate -cp conf_01.gro -p topol_01.top -cs spc216.gro   -o conf_05.gro -maxsol 1909

# Step 06 - Prepare for minimization in water. Maxwarn is set to 1 because otherwise it complains about the non=zero charge (issue fixed below)
gmx_mpi grompp -f mini_wat.mdp -c conf_05.gro -p topol_01.top -po mdp_06.mdp -o mini_water_tmp.tpr -maxwarn 1 

# Step 7 neutralize w NACL ions
echo "SOL" | gmx_mpi genion -neutral -pname NA -nname CL -s mini_water_tmp.tpr -o conf_07.gro -p topol_01.top 

# Step 08 - Prepare for minimization in water (again) 
gmx_mpi grompp -f mini_wat.mdp -c conf_07.gro -p topol_01.top -po mdp_08.mdp -o mini_water.tpr
  
# Step 09 - Run minimization in water#
gmx_mpi mdrun -s mini_water.tpr -deffnm mini_water

#############################################

# Step 10 NVT EQUILIBRATION
gmx_mpi grompp -f equil_nvt.mdp -c mini_water.gro -p topol_01.top -po mdp_10.mdp -o tpr_nvt.tpr
gmx_mpi mdrun -s tpr_nvt.tpr -deffnm nvt 

# step 11 NPT equilibration

gmx_mpi grompp -f equil_npt.mdp -c nvt.gro -t nvt.cpt -p topol_01.top -o tpr_npt.tpr
mpirun -np 4 gmx_mpi mdrun -deffnm npt  -s tpr_npt.tpr 

# step 12 - Prepare production tpr
gmx_mpi grompp -f md.mdp -c nvt.gro -p topol_01.top -po mdp_12.mdp -o chignolin.tpr
