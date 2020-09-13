  
for i in 0.0003 0.001
do
    for j in 64 256
    do
        for k in 0.0003 0.001
        do
            for t in 0.99 0.9
            do
                python train.py -a "${i}" -b "${j}" -c "${k}" -d "${t}"
            done
        done
    done
done