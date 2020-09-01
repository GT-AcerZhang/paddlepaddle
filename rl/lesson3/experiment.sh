for i in 0.0001 0.0003 0.0007
do
    for j in v0 v1 v2
    do
        python train.py --lr "${i}" -m "${j}" 
    done
done

# first compile it chmod +x experiment.sh
# add this to remove unexpected symbol sed -i 's/\r//' experiment.sh
# run it ./experiment.sh