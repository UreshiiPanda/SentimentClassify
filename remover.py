import sys

# usage:   python3 remover.py 1_counts.txt train.txt 1

with open(sys.argv[1], "r") as file1:
    to_remove_words = set(word.strip() for word in file1)

with open(sys.argv[2], "r") as file2, open("train" + str(sys.argv[3]) + ".txt", "w") as output:
    for line in file2:
        words = line.strip().split()
        filtered_words = [word for word in words if word not in to_remove_words]
        new_line = " ".join(filtered_words)
        new_line = new_line[0] + "\t" + new_line[2:] 
        # avoid empty lines
        if new_line != "-" + "\t" and new_line != "+" + "\t":
            output.write(new_line + "\n")


