# Contact Process Simulation

This repository contains a simulation of the Contact Process (CP) in 1+1 dimensions (ring lattice) or on a mean-field all-to-all graph. The Contact Process is a simple model of epidemic spreading or population dynamics that exhibits a phase transition between an active and an absorbing state.

## Parameters

The simulation accepts the following command-line parameters:

### Essential Parameters
- `-l` (float, default=1.1):  
  The infection/contact rate (λ).  
  Critical values:  
  - λ_c ≈ 3.297848 for ring lattice  
  - λ_c = 1 for mean-field (all-to-all) case  

- `-N` (int, default=10000):  
  Number of sites in the network.

- `-M` (int, default=100):  
  Number of memory time steps for quasistationary simulation. When the system enters the absorbing state, it is placed in a random state chosen from the last M visited states. Set to 0 for standard simulation.

### Initial Conditions
- `-X0` (int, default=1):  
  Initial state for sites (0 or 1). Used in combination with `-fX0`.

- `-fX0` (float, default=0.5):  
  Fraction of sites initialized to X0 (remaining sites are set to 0).

- `-noX0Rand` (flag):  
  If set, initial active sites are chosen sequentially rather than randomly.

### Simulation Parameters
- `-tTotal` (int, default=10000):  
  Total number of time steps to simulate.

- `-tTrans` (int, default=0):  
  Number of transient time steps to discard before recording data.

- `-expandtime` (flag):  
  If set (and using sequential update), uses dt=1/N to expand total simulation time: tTotal_eff = tTotal / dt.

### Network and Update Rules
- `-graph` (string, default='ring'):  
  Network topology:  
  - 'alltoall': Mean-field simulation  
  - 'ring': 1+1D with periodic boundary conditions  
  - 'ringfree': 1+1D with free boundaries  

- `-update` (string, default='seq'):  
  Update scheme:  
  - 'seq' or 'sequential': Standard update (1 particle per time step)  
  - 'par' or 'parallel': Attempt to update all particles each time step  

### Simulation Type
- `-sim` (string, default='timeevo'):  
  Simulation type:  
  - 'timeevo': Time evolution (quasistatic if M > 0)  
  - 'aval': Avalanche simulation - seeds 1 site whenever activity dies out  

### Output
- `-outputFile` (string, default='cp.mat'):  
  Name of the output file where results are saved.

## Usage

Run the simulation with command-line arguments, for example:

```bash
python -OO contact_process.py -l 1.001 -graph alltoall -N 1000 -X0 1 -fX0 1 -tTrans 0 -tTotal 1000000 -outputFile cp_mf_lc -update seq
for ($i=0; $i -lt 20; $i++){ python -OO contact_process.py -l 1.001 -graph alltoall -N 1000 -X0 1 -fX0 1 -tTrans 0 -tTotal 1000000 -outputFile cp_mf_lc -update seq; }
for ($i=0; $i -lt 20; $i++){ python -OO contact_process.py -l 1.001 -graph alltoall -N 10000 -X0 1 -fX0 1 -tTrans 1000 -tTotal 10000000 -outputFile cp_mf_lc -update seq; }

# avalanche simulations
python -OO contact_process.py -l 1.00 -graph alltoall -N 10000 -X0 0 -fX0 0 -tTrans 0 -tTotal 50000 -outputFile cp_mf_l1.0_aval -sim aval
python -OO contact_process.py -l 1.00 -graph alltoall -N 10000 -X0 0 -fX0 0 -tTrans 0 -tTotal 1000000 -outputFile cp_mf_l1.0_aval -sim aval

python -OO contact_process.py -l 3.297 -graph ringfree -N 1000 -X0 0 -fX0 0 -tTrans 0 -tTotal 10000 -outputFile cp_ring_l3.297_aval_N1000_t10000 -sim aval
python -OO contact_process.py -l 3.297 -graph ringfree -N 10000 -X0 0 -fX0 0 -tTrans 0 -tTotal 50000 -outputFile cp_ring_l3.297_aval -sim aval
python -OO contact_process.py -l 3.298 -graph ringfree -N 10000 -X0 0 -fX0 0 -tTrans 0 -tTotal 1000000 -outputFile cp_ring_l3.298_aval -sim aval