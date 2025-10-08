linux


1.
txt file，grep ERROR，然后reformat, 创建一个新的log

···
grep "ERROR" app.log > errors_raw.log

awk '/ERROR/ {print $1 "," $2 "," $3 "," substr($0, index($0,$4))}' app.log > errors.csv

sort -u errors_raw.log > errors_sorted.log

grep -v '^#' errors_formatted.log | grep -v '^$' > errors.log
···

2.
ticker symbol和price混在一些，搞成两个column

···
input：AAPL 189.12 MSFT 410.3 GOOGL 168.55 AMZN 198.7
awk '{for (i=1; i<=NF; i+=2) print $i, $(i+1)}' input.txt > output.txt
···

4.
directory里面有很多subdirectory。 Recursively找file，找到所有的lock （包含or exactly）然后改名

···

# 包含 lock 的文件
find /path/to/dir -type f -name '*lock*' -print

# 文件名恰好等于 lock 的文件
find /path/to/dir -type f -name 'lock' -print

假设你想把这些文件名都加个后缀，比如 .bak
find . -type f -name "*lock*" -exec mv {} {}.bak \;

···






