#!/bin/bash

> graph.csv

# run classifiers on variously pruned training data
for prune_num in {1..10}; do
    # generate pruned training data
    cat train.txt | tr -s '[:space:]' '\n' | sort -r | uniq -c | awk -v num="$prune_num" '$1 < num {print $2}' > "$prune_num"_counts.txt
    python3 remover.py "$prune_num"_counts.txt train.txt "$prune_num"
    # compare training results for other classifiers
    # comment out the MLP in classify.py for faster runtimes
    python3 classify-graph.py train"$prune_num".txt dev.txt test.txt | sed "s/$/,$((prune_num - 1))/" >> graph.csv
done


# run the best-performing model on the test data
python3 mnb.py train.txt test.txt > test.txt.predicted 


# file clean up
rm *counts.txt
mv train.txt temp.txt
rm train*.txt
mv temp.txt train.txt



