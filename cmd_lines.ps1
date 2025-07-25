python -OO contact_process.py -l 1.001 -graph alltoall -N 1000 -X0 1 -fX0 1 -tTrans 0 -tTotal 1000000 -outputFile cp_mf_lc -update seq

for ($i=0; $i -lt 20; $i++){ python -OO contact_process.py -l 1.001 -graph alltoall -N 1000 -X0 1 -fX0 1 -tTrans 0 -tTotal 1000000 -outputFile cp_mf_lc -update seq; }

for ($i=0; $i -lt 20; $i++){ python -OO contact_process.py -l 1.001 -graph alltoall -N 10000 -X0 1 -fX0 1 -tTrans 1000 -tTotal 10000000 -outputFile cp_mf_lc -update seq; }



# avalanche simulations
python -OO contact_process.py -l 1.00 -graph alltoall -N 10000 -X0 0 -fX0 0 -tTrans 0 -tTotal 50000 -outputFile cp_mf_l1.0_aval -sim aval
python -OO contact_process.py -l 1.00 -graph alltoall -N 10000 -X0 0 -fX0 0 -tTrans 0 -tTotal 1000000 -outputFile cp_mf_l1.0_aval -sim aval

python -OO contact_process.py -l 3.297 -graph ringfree -N 1000 -X0 0 -fX0 0 -tTrans 0 -tTotal 10000 -outputFile cp_ring_l3.297_aval_N1000_t10000 -sim aval
python -OO contact_process.py -l 3.297 -graph ringfree -N 10000 -X0 0 -fX0 0 -tTrans 0 -tTotal 50000 -outputFile cp_ring_l3.297_aval -sim aval
python -OO contact_process.py -l 3.298 -graph ringfree -N 10000 -X0 0 -fX0 0 -tTrans 0 -tTotal 1000000 -outputFile cp_ring_l3.298_aval -sim aval