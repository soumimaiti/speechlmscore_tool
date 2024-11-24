adir=$1
dest=$2 

mkdir -p ${dest}
for f in ${adir}/*.wav; do
    echo ${dest}/$(basename ${f})
    sox ${f} ${dest}/$(basename ${f}) silence 1 0.1 0.1% reverse silence 1 0.1 0.1% reverse pad 0.2 0.2
done

