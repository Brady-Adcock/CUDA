#!/bin/bash

echo "Removing any jpg files"
rm -f *.jpg

echo "Generating jpg files"
#define an array of image sizes 'width:height'
sizes=('5000:4000' '4000:5000' '5000:5000' '7500:5000' '8000:7500')
for size in ${sizes[@]}; do
   if [[ $size =~ ^([0-9]+):([0-9]+)$ ]]; then
      width="${BASH_REMATCH[1]}";
      height="${BASH_REMATCH[2]}";
      filename="color${width}by${height}.jpg"
      ./generate $filename $width $height
   fi
done

#define an array of block sizes to test 'x:y'
blkSizes=('4:8' '8:8' '16:16' '32:32')

for size in ${sizes[@]}; do
   if [[ $size =~ ^([0-9]+):([0-9]+)$ ]]; then
      width="${BASH_REMATCH[1]}";
      height="${BASH_REMATCH[2]}";
      filename="color${width}by${height}.jpg"
      echo "---------------------";
      for bsize in ${blkSizes[@]}; do
         if [[ $bsize =~ ^([0-9]+):([0-9]+)$ ]]; then
            bw="${BASH_REMATCH[1]}";
            bh="${BASH_REMATCH[2]}";
            echo "./greyscaler -bw $bw -bh $bh $filename"
            ./greyscaler -bw $bw -bh $bh $filename
         fi
      done
   fi
done

echo "Removing jpg files"
rm -f *.jpg
