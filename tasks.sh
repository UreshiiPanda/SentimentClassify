#!/bin/bash

# run classifiers on variously pruned training data
for prune_num in {1..5}; do
    # generate pruned training data
    printf "\n\n"
    printf "Training data with $((prune_num - 1))-count words pruned:"
    cat train.txt | tr -s '[:space:]' '\n' | sort -r | uniq -c | awk -v num="$prune_num" '$1 < num {print $2}' > "$prune_num"_counts.txt
    python3 remover.py "$prune_num"_counts.txt train.txt "$prune_num"
    # get training results for averaged Perceptron
    python3 train-avg.py train"$prune_num".txt dev.txt test.txt
    # compare training results for other classifiers
    # comment out the MLP in classify.py for faster runtimes
    python3 classify.py train"$prune_num".txt dev.txt test.txt
done


# run the best-performing model on the test data
python3 mnb.py train.txt test.txt > test.txt.predicted 


# file clean up
rm *counts.txt
mv train.txt temp.txt
rm train*.txt
mv temp.txt train.txt



