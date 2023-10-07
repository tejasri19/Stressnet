# Stressnet



### Data Pre-processing with Orthomosaic (Raster file) using R libraries

#### To segment ROI (region of interest) from the orthomosaic, we have used R libraries. Our code is intended to process multiple shapefiles within a specified directory and crop a raster image using each shapefile's boundaries. A detailed description of the libraries used is provided below:

1. rgdal, tools and raster are installed.
2. The shapefile is loaded and processed using raster function.
3. This R code appears to be focused on cropping and processing raster data using shapefiles.
4. The raster data is read using raster() amd the shapefile data is read using `readOGR()` and transforms it to match the coordinate reference system (CRS) of the raster data using `spTransform`.
5. The shape (polygon) is used from the shapefile at the current index and stores it in the variable `shp`.
6. The `crop()` function is used to crop the raster data (`dataRaster`) to the extent of the current shape (`shp`).
7. A mask is applied  to the cropped raster using the `mask()` function, keeping only the values within the boundary defined by the shape (`shp`).
8. The cropped and masked raster data are genrated and saved to the specified directory using `writeRaster`.

In summary, this code processes shapefiles in a specified directory, crops a raster image to the extent of each shapefile, and saves the cropped portions as BMP files. The code is specifically designed to work with a single band of raster data and a set number of shapefiles (224 in this case).
