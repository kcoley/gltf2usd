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

## 0.1.5 (2018-11-2)
**Fixed Bugs:**
- Fixed bug in using specular workflow (https://github.com/kcoley/gltf2usd/pull/92)

**Changes:**
- Added alpha mode to Material (https://github.com/kcoley/gltf2usd/issues/88)
- If opacity is set in glTF, overwrite the alpha to 1 on USD export.  
- Alpha mask is not supported in glTF so a warning is displayed and defaults to alpha blend

## 0.1.6 (2018-11-3)
**Fixed Bugs:**
- Fixed bug in joint names to allow support for animations for iOS (https://github.com/kcoley/gltf2usd/issues/79)

## 0.1.7 (2018-11-6)
**Changes:**
- When using alpha opaque, the base color texture is cloned

## 0.1.8 (2018-11-8)
**Changes:**
- If a normal texture has only one channel, the channel is applied to RGB as a new texture

## 0.1.9 (2018-11-10)
**Fixed Bugs:**
- Fixed a bug with transform animations not getting exported

## 0.1.10 (2018-11-12)
**Changes:**
- Added optimize-textures flag to help reduce texture size when generating usdz files

## 0.1.11 (2018-12-4)
**Changes:**
- Set vertex colors for Color4 (vertex color alpha currently not supported)
- Resolves (https://github.com/kcoley/gltf2usd/issues/113)

## 0.1.12 (2018-12-17)
**Changes:**
- Generate a new texture if the KHR_texture_transform extension is used
- Resolves (https://github.com/kcoley/gltf2usd/issues/104)

## 0.1.13 (2018-12-23)
**Fixed Bugs:**
- Fixed a bug where normalized ubytes and ushorts were not being normalized on load
**Changes:**
- Added support for loading embedded images in bufferviews
- Resolves (https://github.com/kcoley/gltf2usd/issues/123)

## 0.1.14 (2018-12-23)
**Fixed Bugs:**
- Fixed a bug where multiple embedded textures within the same buffer were not extracted properly
**Changes:**
- Resolves (https://github.com/kcoley/gltf2usd/issues/125)

## 0.1.15 (2019-01-11)
**Fixed Bugs:**
- Fix for throwing when USD minor version is 19 and path version is 1 [spiderworm](https://github.com/kcoley/gltf2usd/pull/130)
- Fixed bug where texture transform generation would fail on index out of range (https://github.com/kcoley/gltf2usd/issues/131)

## 0.1.16 (2019-01-17)
**Changes:**
- Add option to toggle texture transform texture generation (enable `--generate_texture_transform_texture`, disable `--no-generate_texture_transform_texture`) (https://github.com/kcoley/gltf2usd/issues/133)

## 0.1.17 (2019-01-24)
**Changes:**
- Preserve material names of gltf if present (https://github.com/kcoley/gltf2usd/issues/135)

## 0.1.18 (2019-01-28)
**Changes:**
- Cache accessor data to help with glTF import optimization (https://github.com/kcoley/gltf2usd/issues/137)
- Adding logging for exporting mesh primitive data, behind the -v flag

## 0.1.19 (2019-01-30)
**Changes:**
- Prevent crash and display warning when glTF nodes have joints, but no skin (https://github.com/kcoley/gltf2usd/issues/140)

## 0.1.20 (2019-02-04)
**Fixed Bugs:**
- Fixed a bug where textures were being indexed through images (https://github.com/kcoley/gltf2usd/issues/142)

## 0.1.21 (2019-02-11)
**Changes:**
- Update rename regex to include square brackets (https://github.com/kcoley/gltf2usd/issues/144)

## 0.1.22 (2019-02-15)
**Fixed Bugs:**
- Resolve some issues with materials containing special characters by appending index to name (https://github.com/kcoley/gltf2usd/issues/147)

## 0.1.23 (2019-02-27)
**Fixed Bugs:**
- Fixed a bug where `TEXCOORD_1` attribute was not being read (https://github.com/kcoley/gltf2usd/issues/149)