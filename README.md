# Valorant analytics


# Usage

## maps

make png files from video

```
python valo/video/video2img.py <video.mp4> <output_dir>
```

extract map image

```
python valo/video/mapmatcher.py -t <template.png> -i <image_dir> -o <output_dir>
```

make video from png files

```
 ffmpeg -r 24 -i "<input_dir>/video_%04d.png" -vcodec libx264 -pix_fmt yuv420p <output_file>
```