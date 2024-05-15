export OMP_NUM_THREADS=5
for var in {0..9}
do
    {
    cd ../$var
    python -u main.py > log
    }&
done
