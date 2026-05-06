STEP 1:
* Open Image Directory and Locate Ideal Image for Calibration (Average Surface Terrain) *
~ Ideal Image should be without of Cracks, Holes, Wetspots, or Gravel/Debris ~

########################################
### Run Interactive Script for Image ###
########################################
python main.py \
    --image {IMAGE-PATH} \
    --interactive

STEP2:
* Select GUI window and key "q" *
* Then reference the max values for the threshold values *

STEP 3:
########################################
### Run Annotation Script for Videos ###
########################################
python main.py \
  --video {VIDEO-PATH}\
  --max-contrast {MAX-CONTRAST-THRESHOLD} \
  --max-entropy {MAX-ENTROPY-THRESHOLD} \
  --max-std-contrast {MAX-STD-CONTRAST-THRESHOLD} \
  --min-homogeneity {MIN-HOMOGENEITY-THRESHOLD} \
  --min-correlation {MIN-CORRELATION-THRESHOLD} \
  --min-energy {MIN-ENERGY-THRESHOLD}
