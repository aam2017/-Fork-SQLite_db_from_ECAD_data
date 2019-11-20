##############################
# Merge PNG images into GIF. #
##############################

mogrify -resize 500x400 *.png

convert -delay 20 -loop 0 *.png weather.gif
