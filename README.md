# TODO


## Retrieval scripts
- [ ] Check why the bounding boxes of the labeled areas produced by Roberto don't include all the parcels. 
  It might be because the automatation script does not take into account parcel with different naming patterns.
- [ ] Make the statistics of the ground truth.
- [ ] Use information for different seassons of the year.
- [ ] Add the computation of NDVI and other parameters (as explained in Catherine's document) as an option after the retireval of Sentinel 2 images. 
  Ideally also for sentinel 1 images after Roberto's inference model is well tunned.

- [ ]For all the script that require input file:
  - [ ] Create a list
    - [ ] Containing the path to the file
    - [ ] Field with the syntax or format explanation.
  - [ ] When needed possible create a list maker.
- [ ] implement classes for the retrieval scripts.

## Binary classification
- [ ] Consider that the labeling of the the pixel outside the polygon and inside the bounding box as non vinard could carry a big error, 
  as they might be viniard from a non selected parcel.
- [x] Add the input files lists
- [x] Run in Recas for more data.
- [x] Implement a more complex model.
- [ ] Implement patches processing.
  - [ ] Make some research to look for different possibilities.
- [ ] Implement validation.
- [ ] Implement test.
- [ ] Implement inference.
- [ ] Make some nice plots.

# Agro_intesa
## Snipets and scripts for the Agro@intesa project

#### Syntax for using the converter:

```
./conversion_kmz2geojson.py -i feudi_21256.kmz
```
