#test convert to openvisus
import convert
import glob

RGB_DIR = "/Users/ben/Dropbox/Weecology/NEON_Visualization/OSBS/rgb/"
OUTDIR = "/Users/ben/Dropbox/Weecology/NEON_Visualization/OSBS/output/"
ANNOTATION_DIR = "/Users/ben/Dropbox/Weecology/NEON_Visualization/OSBS/predictions/"

rgb_images = glob.glob(RGB_DIR + "*.tif")

#local data not on travis
def test_run():
    convert.run(rgb_images=rgb_images, annotation_dir=ANNOTATION_DIR, outdir=OUTDIR)