from pathlib import Path

OUT_NUM = 5

path = Path().resolve()
logpath = (path / 'logs/pycov/30/log.txt')
out_files = []


for i in range(OUT_NUM):
    out = (path/ 'logs/pycov/30/log{}.txt'.format(i))
    out_files.append(out)

# Read input file
with open(str(logpath), 'r') as f:
    lines = f.readlines()

s = int(len(lines)/OUT_NUM)             # lines in each file

for i in range(OUT_NUM-1):
    with open(out_files[i], 'w') as f:
        f.writelines(lines[s*i:s*(i+1)])

# write last file with left lines
with open(out_files[-1], 'w') as f:
    f.writelines(lines[(OUT_NUM-1)*s:])


