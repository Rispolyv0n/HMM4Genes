#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
start: "ttggtaccat"
end: "CTTTGCCTG"
'''


file_path = "./data/GRCh38_latest_genomic.fna"

with open(file_path, 'r') as f:
    start = False
    cc = ""

    for line in f:
        if start:
            if ">" in line:
                break
            else:
                cc += line
        if ">NC_000006.12 Homo sapiens chromosome 6" in line:
            start = True

cc = cc.split("\n")
cc = "".join(cc)
cc_train = cc[100000:200000]
cc_test = cc[200001:300001]

with open("./NC_000006_12_Homo_sapiens_chromosome_6_GRCh38_p13_Primary_Assembly.txt", 'w') as c:
    c.write(cc_train)

with open("./NC_000006_12_Homo_sapiens_chromosome_6_GRCh38_p13_Primary_Assembly_test.txt", 'w') as c:
    c.write(cc_test)
