#Changelog

## 0.1.0 (2018-10-15)
 **Fixed Bugs:**
  - Fixed an issue where generating `usdz` wrote out absolute paths instead of relative paths 

**Changes:**
 - Reorganized repo by moving files to the `_gltf2usd` module
 - Renames unicode filenames to ascii to avoid name conflicts (https://github.com/kcoley/gltf2usd/issues/71)
 - Added versioning to the `gltf2usd` (https://github.com/kcoley/gltf2usd/issues/81)
(https://github.com/kcoley/gltf2usd/issues/68)
 - Added changelog
 

 ## 0.1.1 (2018-10-21)
 **Fixed Bugs:**
 - Fixed issue with node transforms and node animations, where the glTF node transform would get applied on top of its animation, causing incorrect positions.
 
 **Changes:**
 - Convert usd file path to absolute path on load  
 - Reformatted changelog
 - Switched some methods to properties
 - Increment version
 
## 0.1.2 (2018-10-21)
**Fixed Bugs:**
- Use only first animation group

## 0.1.3 (2018-10-26)
**Fixed Bugs:**
- Fixed strength

## 0.1.4 (2018-10-28)
**Fixed Bugs:**
- Opacity for base color factor is now used even wihout base color texture