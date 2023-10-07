## Title   :: Cropping data using shape files
#  Author  :: Tejasri
#  DOC     :: 20221020
#  DOLE    :: 20221020
#  Remarks :: 

## ClearUp and dir ####################################################
graphics.off(); rm(list = ls()); cat("\014")
setwd("/home/tejasri/Segmented Plots/Season_4_Paddy_Segmented_Plots/paddy_season4_RededgeMultispectral_20191024_05m_Blue")
## Load required libraries ############################################
if(!require(rgdal)){install.packages("rgdal");library(rgdal)}
if(!require(tools)){install.packages("tools");library(tools)}
if(!require(raster)){install.packages("raster");library(raster)}


## Main code ##########################################################
WorkFolders <- dir() # get all the folder in root directory for automation
CurrentFolder <- paste(getwd(), '/',WorkFolders[0], sep = "")


ListDir <-  list.files(path = CurrentFolder, full.names = FALSE, recursive = TRUE)
for(ii in 1: length(ListDir)){
  if(file_ext(ListDir[ii]) == "shp")
  {
    shapeName <- substr(ListDir[ii],1, nchar(ListDir[ii])-4) 
  }
}

for(Band in 1:1){
  dataRaster <- raster('./paddy_season4_RededgeMultispectral_20191024_05m_Blue.tif', band = Band)
  #dataRaster <- spTransform(dataRaster, CRS("+init=epsg:24383"))
  
  ShpCollage <- readOGR(CurrentFolder, shapeName)
  ShpCollage <- spTransform(ShpCollage,crs(dataRaster))
  
  for(ii in 1:224){
    shp <- ShpCollage[ii,]
    r2 <- crop(dataRaster, extent(shp))
    r3 <- mask(r2, shp)
    writeRaster(r3,filename = paste(CurrentFolder,'/Cropped/','Paddy_Blue_season4_20191024_05m_Plot_',ii,'.bmp', sep = ""), format= 'BMP', overwrite=TRUE)
    print(ii)
  }
}
