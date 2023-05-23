

# CATEGORIES="bpd_multi bpd_bi1 bpd_bi2 bpd_bi3 rds1_multi rds2_multi rds1_bi1 rds1_bi2 rds1_bi3 rds2_bi1 rds2_bi2 rds2_bi3"
CATEGORIES="bpd_bi1 bpd_bi2 bpd_bi3 rds1_bi1 rds1_bi2 rds1_bi3 rds2_bi1 rds2_bi2 rds2_bi3"


for category in $CATEGORIES
do
    echo $category
    python table_total.py --target $category
done
