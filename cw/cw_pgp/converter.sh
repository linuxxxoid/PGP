
# usage: bash converter.sh [path-to-data-images] [path-to-save-jpgs]

mkdir -p $2
rm $2/*
for file in $1/*
do 
    f=${file#$1/};
    echo Converting $file to $2/${f%.data}.jpg
    python2 conv.py $file $2/${f%.data}.jpg 
done