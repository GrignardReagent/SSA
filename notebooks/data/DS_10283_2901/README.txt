README FILE MAY 21, 2018

Data from single-cell microscopy experiments for Granados, Pietsch,
Cepeda-Humerez, Farquhar, Tkacik and Swain (2018) Distributed and dynamic
intracellular organization of extracellular information in yeast. PNAS.
(https://dx.doi.org/10.1073/pnas.1716659115)


Explanation of the storage format
---------------------------------

The data from each group of experiments have been exported in JSON (JavaScript
Object Notation) format (RFC 7159; https://tools.ietf.org/html/rfc7159.html).

The files are text-based and use the JSON format to specify a structured
hierarchy of named data matrices. Each matrix is specified as an Array of
Arrays of numbers, where the rows (outer Array) distinguish cells and the
columns (inner Arrays) distinguish time points for the given measurement and
experiment. If a cell was not present at a given time point, that measurement
is annotated using the 'null' literal.

At the end of this README you can find examples demonstrating valid import of
the data into some popular scripting languages for data analysis.


Description of the data
-----------------------

This repository contains data extracted from segmented time-lapse fluorescence
microscopy images of yeast grown in ALCATRAS microfluidics devices [1], with
cells segmented using the DISCO algorithm [2], an active contour method based
on the brightfield image.

Each file contains the data for a collection of related experiments, organised
in a hierarchy as described below for each file. In all cases, the leaf nodes
of the JSON Object hierarchy contain matrices labelled by the measurement
type which may include some or all of the following:

- median: the median fluorescence intensity of pixels in the segmented area;
- max5: the mean fluorescence intensity of the 5 brightest pixels in the
  segmented area;
- nucLoc: the ratio of max5 to median (i.e., max5/median);
- nucLocNorm: nucLoc normalized as described in supplementary text;
- origin: the time point at which media switching was determined to occur;
- times: the time at which the image was taken, relative to the origin; and
- imBackground: the median pixel intensity for pixels not included in any
  segmented areas.

File descriptions:

- fig1_sfp1_replicates.json
  Data from experiments pertaining to Fig. 1 of the main text and Fig. S7 of the
  supplementary text for nuclear localization of Sfp1 in response to
  environmental transitions from rich media to carbon stress (0.1% glucose).
  The data are for six biological replicates performed on different days, and
  are grouped first by replicate (with labels 'rep1', 'rep2', ..., 'rep6'), and
  then by channel name and measurement type (the 'general' channel containing
  'origin' and 'times', the 'GFP' channel containing 'median', 'max5', 'nucLoc'
  and 'imBackground', and the 'cy5' channel containing 'imBackground').

- fig2_stress_type_expts.json 
  Data from experiments pertaining to Figs. 2B-D, 3A-B and 4B-D of the main
  text, and Figs. S4, S6A, S8A, S9, S11, S15, S17A, S18A-D, S20, S21 and S22A of
  the supplementary text for environmental transitions from rich media to either
  carbon stress (0.1% glucose), osmotic stress (0.4 M NaCl) or oxidative stress
  (0.5 mM H2O2). Note that this data is simply that for the first replicate in
  'fig2_stress_type_replicates.json' (see below), except that more measures
  extracted from the images are included. The data are grouped first by strain
  (labeled according to the tagged transcription factor: 'msn2', 'msn4', 'dot6',
  'tod6', 'sfp1', 'maf1', 'mig1', 'mig2', 'hog1' or 'yap1'), then grouped by
  stress type ('oxid' for oxidative stress, 'gluc' for carbon stress and 'nacl'
  for osmotic stress), and then by channel name and measurement type (as per
  'fig1_sfp1_replicates.json').

- fig2_stress_type_replicates.json 
  As for 'fig2_stress_type_expts.json' (see above), except the data includes
  an additional biological replicate, and in this case, only the normalized
  nuclear localization described in the supplementary text is included (see
  'fig2_stress_type_expts.json' for other measurements of the first replicate).
  As per 'fig2_stress_type_expts.json', the data are first grouped by strain and
  then stress type, but this is followed by additional grouping into replicate
  (either 'rep1' or 'rep2'), before grouping by channel name and measurement
  type (the 'GFP' containing 'nucLocNorm' and 'origin').

- fig3_stress_level_expts.json
  Data from experiments pertaining to Figs. 3C-E and 4D of the main text,
  and Figs. S6B, S8B, S10, S12, S13, S14B, S17B, S18E, S22B, and S23 of the
  supplementary text for environmental transitions from rich media to
  different magnitudes of either carbon, osmotic or oxidative stress. The data
  are grouped first by stress type ('gluc' for carbon stress, 'nacl' for
  osmotic stress and 'oxid' for oxidative stress), then by strain (labeled
  according to the tagged transcription factor: 'msn2', 'dot6', 'sfp1' and
  'maf1' for all stress types, but 'mig1', 'hog1' and 'yap1' only for their
  cognate stress), then by stress magnitude (for carbon stress, a transition
  to 0.8% glucose is labeled 'gluc_0_8'; for osmotic stress, a transition to
  0.1 M NaCl is labeled 'nacl_0_1M'; for oxidative stress, a transition to
  0.22 mM hydrogen peroxide is labeled 'oxid_0_22mM'), and then by channel
  name and measurement type (as per 'fig1_sfp1_replicates.json').

- 'figS2_nuclear_marker_expts.json'
  Experiments pertaining to Figs. S2 and S5 of the supplementary text for the
  environmental transition from rich media to low glucose (0.1%). The data are
  first grouped by strain (named according to the transcription factor labeled
  by GFP), then by channel (GFP), and then by the extracted measurement. This
  data set also included fluorescent images of a nuclear marker
  (Nhp6A-mCherry), and watershed segmentation of the Nhp6A tag was used to
  provide the following additional measures:
  - cytInnerMedian: the median fluorescence intensity of pixels in the
    segmented cell area that are not in the segmented nucleus; and
  - nucInnerMedian: the median fluorescence intensity of pixels in the
    segmented nucleus.

- 'figS8_rich_to_rich_expts.json'
  Experiments pertaining to Fig. S8 of the supplementary text for the
  environmental transition from rich media to rich media. In preparing
  the figure, this data was supplemented with the data contained in
  'fig2_stress_type_expts.json'. The data are first grouped by strain (named
  according to the transcription factor labeled by GFP), then by channel and
  measurement (as per 'fig1_sfp1_replicates.json').

- 'figS19_two_colour_expts.json'
  Experiments pertaining to Fig. S19 of the supplementary text for strains
  with two fluorescent tags undergoing environmental transitions from rich
  media to either carbon stress (0.1% glucose) or osmotic stress (0.4 M NaCl).
  The data are grouped first by strain (labeled according to the tagged
  transcription factors, e.g., 'dot6_msn2', where the first, Dot6, is tagged
  using GFP and the second, Msn2, using mCherry), then grouped by stress type
  ('gluc' for carbon stress and 'nacl' for osmotic stress), and then by
  channel name (either GFP or mCherry) and measurement type (as per
  'fig1_sfp1_replicates.json').


References:

[1] Crane MM, et al. (2014) A microfluidic system for studying ageing and
    dynamic single-cell responses in budding yeast. PLoS One, 9(6):e100042.
[2] Bakker E, Swain PS, Crane MM (2018) Morphologically constrained and data
    informed cell segmentation of budding yeast. Bioinformatics 34:88-96.
    Source code at https://github.com/pswain/DISCO.


Example of data import in Python (https://www.python.org/)
----------------------------------------------------------

The `json` module is a standard library that ships with Python, so no
additional modules are required to load the data. The example below, however,
makes use of NumPy (http://www.numpy.org/) and Matplotlib
(https://matplotlib.org/) for analysis and plotting.

>>> import json
>>> with open('figS1_nuclear_marker_expts.json','r') as f:
...     expts = json.load(f)
...
>>> import numpy as np
>>> import matplotlib.pyplot as plt
>>> times = np.array(expts['dot6']['GFP']['times'], dtype='float')
>>> nucLoc = np.array(expts['dot6']['GFP']['nucLoc'], dtype='float')
>>> plt.plot(np.mean(times,0), np.mean(nucLoc,0))
>>> plt.xlabel('Time (min.)')
>>> plt.ylabel('Nuclear localization')


Example of data import in R (https://www.r-project.org/)
--------------------------------------------------------

Import of JSON files requires the `jsonlite` package, which can be obtained by
running:

> install.packages('jsonlite')

Once the package is installed, the data sets can be loaded as:

> library(jsonlite)
> expts = fromJSON('figS1_nuclear_marker_expts.json')
> plot(apply(expts$dot6$GFP$times,2,mean), # mean time across cells
       apply(expts$dot6$GFP$nucLoc,2,mean), # mean nuclear localization
       type='l', xlab='Time (min.)', ylab='Nuclear localization')


Example of data import in Matlab (https://www.mathworks.com/)
-------------------------------------------------------------

Import of JSON files requires the JSONlab toolbox
(http://iso2mesh.sf.net/cgi-bin/index.cgi?jsonlab), which can be obtained from
the Mathworks File Exchange
(http://mathworks.com/matlabcentral/fileexchange/33381).

Once the toolbox is on your Matlab path, the data sets can be loaded as:

>> file_string = fileread('figS1_nuclear_marker_expts.json')
>> expts = loadjson(regexprep(file_string,'null','NaN'));
>> plot(mean(expts.dot6.GFP.times), mean(expts.dot6.GFP.nucLoc))
>> xlabel('Time (min.)');
>> ylabel('Nuclear localization');
